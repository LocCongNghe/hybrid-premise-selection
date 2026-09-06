import Mathlib
import Qq

open Lean Elab Command Meta

/-!
# Lean kernel applicability reranker probe (I3, Idea 1)

For premise selection, the strongest *semantic* signal of whether a candidate theorem
`c` is relevant to a query term `e` is a kernel-level check: **can the conclusion of
`c`'s statement (with its universal binders turned into metavariables) be unified
(`isDefEq`) with the type of `e`?**

This is orthogonal to all similarity signals used so far (WL/TEDS = tree structure,
dense = text cosine, BM25 = lexical). In particular it can break the structural ties
that defeat dense-in-fusion (the Test B R@1 wall): a candidate with identical tree
structure to the target but a non-unifiable conclusion is *definitionally* not
applicable, regardless of cosine.

## Probe definition

For a query term `e` (Test A: a Lean term; its elaborated type is `T_q`) and a
candidate theorem named `c` with statement type `T_c = ∀ x1..xn, C`:

  1. Peel the leading `forallE`/`letE` binders of `T_c`.
  2. Replace the peeled binder-domain dependencies with fresh metavariables
     (`mkForallFVars`' universe of binders → `mkAppM`'s of fresh `mvar`s), yielding
     the *conclusion* `C'` with metavariable holes where the binders were.
  3. `isDefEq T_q C'` → `applicable := true` if the unifier can make the candidate's
     conclusion definitionally equal to the query's type.

A refinement (`mode`): full `isDefEq` on the whole conclusion may be too strict
(many relevant premises have conclusions that unify only after some unfolding).
Three modes are offered:
  - `full`   : `isDefEq T_q C'` on the entire conclusion (strictest).
  - `head`   : `isDefEq` on the *head* of `T_q` vs head of `C'` after peeling, i.e. do
               they share a definitional head constant. Much looser, much cheaper.
  - `whnf`   : reduce both `T_q` and `C'` to WHNF, then `isDefEq` on the WHNF heads.

## I/O

Input  (requests file, TSV):  `query_idx \t query_term \t cand1,cand2,...,candN`
Output (results  file, TSV):  `query_idx \t cand_name \t applicable \t mode`

Failures (parse/elab/not-found) are emitted with `applicable=` empty so no record is
lost; the driver skips them. Processing is batched (`liftTermElabM` per query) to
bound Meta-state growth (same lesson as CleanPP).

Usage (from the `lean/` dir):
  lake env lean --run TBPS/KernelProbe.lean full  REQUESTS.tsv RESULTS.tsv
  lake env lean --run TBPS/KernelProbe.lean head   REQUESTS.tsv RESULTS.tsv
  lake env lean --run TBPS/KernelProbe.lean whnf   REQUESTS.tsv RESULTS.tsv
-/

namespace TBPS.KernelProbe

/-- Strip the Qq `q(...)` quotation wrapper (mirrors lean_extractor). -/
def stripQqWrapper (s : String) : String :=
  let t := s.trim
  if t.startsWith "q(" && t.endsWith ")" then (t.drop 2).dropRight 1 |>.trim else t

/-- Parse a dotted Lean name string into a `Name`. -/
def parseNameStr (s : String) : Name :=
  (s.split (· == '.')).foldl (fun acc p => if p.isEmpty then acc else Name.str acc p) Name.anonymous

/-! ## JSON → Expr deserializer (for Test B `state` payloads)

The Test B manifest carries each query's goal type as a `state` JSON tree in the same
12-constructor single-key-dict format the Python `deserialize_expr` parses. We deserialize
it directly into a kernel `Expr` here, with one key adaptation: every `fvar` (a free
variable of the goal context, identified by a placeholder string id like
``Lean.Name.mkNum`_uniq1420``) becomes a fresh **metavariable**, and each `forallE` binder
is peeled by substituting its de-Bruijn body reference with another fresh metavar. The
result is exactly the *generalized conclusion* we want to feed `isDefEq` — no elaboration,
no free-variable context needed. This mirrors what `conclusionWithMVars` does for an
already-elaborated Expr. -/

/-- Parse a Lean universe-string (e.g. ``"Lean.Level.zero"`` or ``"u"``) into a `Level`.
    We only ever see `zero` and param names in practice; unknown → a named param. -/
def parseLevelStr (s : String) : Lean.Level :=
  match s with
  | "Lean.Level.zero" | "0" => .zero
  | _ => .param s.toName

/-- Get the single (key, value) pair of a constructor object, or fail.
    We probe known constructor keys with `getObjValD` (returns null if absent) rather
    than iterating the object's RBNode, to avoid node-traversal API friction. -/
def jsonCtor (j : Lean.Json) : MetaM (String × Lean.Json) := do
  let ctors := ["const","app","fvar","bvar","mvar","sort","forallE","lam","letE","lit","mdata","proj"]
  let mut found : Option (String × Lean.Json) := none
  for c in ctors do
    let v := j.getObjValD c
    if !v.isNull then found := some (c, v)
  match found with
  | some kv => return kv
  | none => throwError s!"jsonCtor: no known constructor key in {j.compress}"

/-- `getObjValD` as a convenience (returns null Json if missing). -/
def jfield (j : Lean.Json) (k : String) : Lean.Json := j.getObjValD k

/-- `getStr?` unwrapped (empty string if missing/error). -/
def jstr (j : Lean.Json) (k : String) : String :=
  match (j.getObjValD k).getStr? with | .ok s => s | _ => ""

/-- Deserialize a `state` JSON Expr into a kernel `Expr` with fvars → metavars.
    `ref` holds the fvarId-string → metavar map so the same id resolves consistently. -/
partial def parseExprJson (j : Lean.Json) (ref : IO.Ref (Std.HashMap String Expr)) : MetaM Expr := do
  let (ctor, val) ← jsonCtor j
  match ctor with
  | "const" =>
    let name := parseNameStr (jstr val "declName")
    let us := match (val.getObjValD "us").getArr? with | .ok a => a | _ => #[]
    let levels := us.map (fun u => parseLevelStr (match u.getStr? with | .ok s => s | _ => "0"))
    return .const name levels.toList
  | "app" =>
    let fn ← parseExprJson (jfield val "fn") ref
    let arg ← parseExprJson (jfield val "arg") ref
    return .app fn arg
  | "fvar" =>
    -- map placeholder fvarId → one fresh metavar per distinct id (consistent across refs)
    let id := jstr val "fvarId"
    let m ← ref.get
    match m.get? id with
    | some mv => return mv
    | none =>
      let mv ← mkFreshExprMVar none
      ref.set (m.insert id mv)
      return mv
  | "bvar" =>
    -- de-Bruijn index; kept as a bvar, resolved when the enclosing forallE is peeled
    let s := jstr val "deBruijnIndex"
    match s.toNat? with | some n => return .bvar n | none => return .bvar 0
  | "mvar" =>
    let mv ← mkFreshExprMVar none
    return mv
  | "sort" =>
    let u := jstr val "u"
    return .sort (parseLevelStr u)
  | "forallE" =>
    -- peel: parse the body (with its bvar-0 reference), substitute bvar 0 with a fresh
    -- metavar. binderType is parsed only for safety (free metavars may appear in it too).
    let _ ← parseExprJson (jfield val "binderType") ref
    let body ← parseExprJson (jfield val "body") ref
    let mv ← mkFreshExprMVar none
    return body.instantiate1 mv
  | "lam" =>
    let _ ← parseExprJson (jfield val "binderType") ref
    let body ← parseExprJson (jfield val "body") ref
    let mv ← mkFreshExprMVar none
    return body.instantiate1 mv  -- treat like an existential: λ-bound var → metavar
  | "letE" =>
    -- simplify: substitute let-bound var with its value (let unfolding)
    let _ ← parseExprJson (jfield val "type") ref
    let v ← parseExprJson (jfield val "value") ref
    let body ← parseExprJson (jfield val "body") ref
    return body.instantiate1 v
  | "lit" =>
    let s := jstr val "literal"
    match s.toNat? with
    | some n => return .lit (.natVal n)
    | none => return .lit (.strVal s)
  | "mdata" =>
    -- discard metadata, keep the inner expr
    parseExprJson (jfield val "expr") ref
  | "proj" =>
    let tname := parseNameStr (jstr val "typeName")
    let idx := match (val.getObjValD "idx").getNat? with | .ok n => n | _ => 0
    let struct ← parseExprJson (jfield val "struct") ref
    return .proj tname idx struct
  | _ => throwError s!"parseExprJson: unknown constructor {ctor}"

/-- Deserialize a `state` JSON string into a generalized-conclusion `Expr`
    (fvars/forall-binders → metavars). -/
def parseStateJson (s : String) : MetaM Expr := do
  match Lean.Json.parse s with
  | .error e => throwError s!"parseStateJson: JSON parse error: {e}"
  | .ok j =>
    let ref ← IO.mkRef (Std.HashMap.empty : Std.HashMap String Expr)
    parseExprJson j ref

/-- Elaborate a raw Test A term string to a fully-instantiated `Expr`. -/
def elabTermStr (env : Environment) (s : String) : TermElabM Expr := do
  let stripped := stripQqWrapper s
  match Lean.Parser.runParserCategory env `term stripped with
  | .error err => throwError s!"parse error: {err}"
  | .ok stx =>
    let e ← Elab.Term.elabTerm stx none
    instantiateMVars e

/-- Result of peeling the leading `forallE`/`letE` binders of a type expression:
    the list of (binderName, fvar) for the peeled binders, and the body Expr. -/
partial def peelForall (type : Expr) : MetaM (Array (Name × Expr) × Expr) := do
  let mut binders : Array (Name × Expr) := #[]
  let mut e := type
  let mut guard := 0
  while guard < 64 do
    guard := guard + 1
    match e with
    | .forallE n d b _ =>
      -- introduce a fresh fvar for this binder, recurse on the instantiated body
      let fvar ← mkFreshFVarId
      let fv := Expr.fvar fvar
      binders := binders.push (n, fv)
      e := b.instantiate1 fv
    | .letE n t v b _ =>
      let fvar ← mkFreshFVarId
      let fv := Expr.fvar fvar
      binders := binders.push (n, fv)
      e := b.instantiate1 fv
    | _ => break
  return (binders, e)

/-- Turn the peeled binders into fresh metavariables and apply them to the body's
    conclusion, returning the conclusion Expr with metavariable holes.
    For a theorem `∀ x1..xn, C`, returns `C` with `?m1..?mn` substituted where the
    binders were, i.e. a conclusion whose free holes are the universal variables. -/
def conclusionWithMVars (type : Expr) : MetaM Expr := do
  let (binders, body) ← peelForall type
  if binders.isEmpty then return body
  -- Create one fresh mvar per peeled binder (domain = the binder's domain, which for
  -- forallE is the binder type). We don't need dependent domains to be perfect;
  -- mkFreshExprMVar gives a fresh mvar of the given type.
  let mut mvars : Array Expr := #[]
  -- Build mvars left-to-right; for dependent binders a later domain may mention an
  -- earlier binder, but we already substituted fvars, so domains reference fvars that
  -- no longer exist. To keep it simple and robust, we substitute fvars -> mvars as we
  -- go by collecting the (fvar, mvar) pairs and abstracting at the end.
  let mut pairs : Array (FVarId × Expr) := #[]
  for (n, fv) in binders do
    -- fv is Expr.fvar id; its domain was lost in peelForall (we only kept the body).
    -- We use `mkFreshExprMVar none` (any type) — the unifier will assign types as it
    -- unifies the conclusion with T_q. This is the standard "existential" reading.
    let mvar ← mkFreshExprMVar none
    mvars := mvars.push mvar
    match fv with
    | .fvar id => pairs := pairs.push (id, mvar)
    | _ => pure ()
  -- Substitute fvars -> mvars in the body. replaceFVarId takes an FVarId directly.
  let body' := pairs.foldl (fun acc (id, mvar) => acc.replaceFVarId id mvar) body
  return body'

/-- The head constant name of an Expr after WHNF, if any. -/
def headConstAfterWhnf (e : Expr) : MetaM (Option Name) := do
  let w ← whnf e
  match w.getAppFn with
  | .const n _ => return some n
  | _ => return none

/-- Core applicability check for one (queryExpr, candidateName) pair.
    Returns `some true` / `some false` if the check ran; `none` on error.

    NB on the match: Test A query terms elaborate to PROPOSITIONS (type `Prop`), so
    `inferType queryExpr` would be `Prop` for every query → useless (every theorem
    conclusion is also `Prop` → always true, no discrimination). Instead we match the
    *statement bodies* directly: peel the query's own implication/forall binders and
    generalize them to metavars (same treatment as the candidate), then `isDefEq` the
    two generalized bodies. The target theorem will match its own query; the question
    is how many *other* candidates share the generalized shape. -/
def checkApplicable (env : Environment) (mode : String) (queryExpr : Expr)
    (candName : Name) : TermElabM (Option Bool) := do
  match env.find? candName with
  | none => return none  -- candidate not in environment (e.g. core/Batteries name)
  | some info =>
    let candType : Expr := info.type
    -- Bounding pathological isDefEq / reduction. A candidate whose conclusion (or a query
    -- goal) is deeply nested can hang `isDefEq`, `whnf`, or even `conclusionWithMVars`
    -- (peeling + `replaceFVarId` over a huge body) for minutes. The bound works in TWO parts:
    --   (1) FIRE: a per-call heartbeat ceiling. `checkMaxHeartbeats` reads
    --       `(← read).maxHeartbeats` from Core.Context (captured at creation; NOT refreshed
    --       by `withOptions`/`withTheReader` in TermElabM). So the ceiling must be set
    --       GLOBALLY via `set_option maxHeartbeats` in the scratch driver (which sets the
    --       captured value). `withCurrHeartbeats` resets `initHeartbeats` so each
    --       checkApplicable call gets a fresh delta (no cross-call exhaustion). Verified:
    --       the ceiling FIRES (the "(deterministic) timeout, 8000000 heartbeats" error
    --       appears). `withTheReader Core.Context` is kept as belt-and-suspenders.
    --   (2) CATCH: `tryCatchRuntimeEx` catches the `runtime.maxHeartbeats` throw (an
    --       `Exception.error` tagged `runtime.maxHeartbeats` — NOT an interrupt, so the
    --       CoreM `tryCatchRuntimeEx` catches it; verified empirically inside `liftTermElabM`).
    --       The catch MUST wrap the ENTIRE body — `conclusionWithMVars`, `whnf`, AND
    --       `isDefEq` — because the heartbeat can fire in ANY of them (an earlier version
    --       wrapped only `isDefEq`; a pathological `conclusionWithMVars`/`whnf` then threw
    --       past the catcher, and `liftTermElabM`'s `observing`+`ofExcept` re-threw it at
    --       the command level, aborting the driver's `for` loop before the 5-query flush).
    --       On catch we return `none` (treat the pair as unknown / not-boosted).
    -- `withReducible` additionally restricts unfolding to `[reducible]` defs.
    let hbLimit : Nat := 8000000
    Lean.withCurrHeartbeats do
      withTheReader Core.Context (fun ctx => { ctx with maxHeartbeats := hbLimit }) do
        Lean.Meta.withReducible do
          tryCatchRuntimeEx
            (do
              -- generalize the QUERY body (peel its binders -> metavars)
              let qBody ← conclusionWithMVars queryExpr
              match mode with
              | "full" =>
                let concl ← conclusionWithMVars candType
                let ok ← isDefEq qBody concl
                return some ok
              | "whnf" =>
                let concl ← conclusionWithMVars candType
                let qw ← whnf qBody
                let cw ← whnf concl
                let ok ← isDefEq qw cw
                return some ok
              | "head" =>
                let concl ← conclusionWithMVars candType
                match (← headConstAfterWhnf qBody), (← headConstAfterWhnf concl) with
                | some a, some b => return some (a == b)
                | _, _ => return some false
              | _ => return none)
            (fun _ => return none)  -- heartbeat / runtime exception -> unknown

end TBPS.KernelProbe

/-- Driver command. Reads REQUESTS (TSV: qidx \t term \t c1,c2,...) and writes
    RESULTS (TSV: qidx \t cand_name \t applicable \t mode) per (query,candidate). -/
elab (name := tbpsKernelProbe) "tbps_kernel_probe " mode:str reqFile:str resFile:str : command => do
  let mode := mode.getString.trim
  let reqText ← IO.FS.readFile reqFile.getString
  let lines := reqText.split (· == '\n') |>.filterMap (fun l => let t := l.trim; if t.isEmpty then none else some t)
  logInfo m!"[KernelProbe:{mode}] {lines.length} query records from {reqFile.getString} → {resFile.getString}"
  let env ← getEnv
  let out ← IO.FS.Handle.mk resFile.getString .write
  let mut buf : String := ""
  let mut qOk := 0
  let mut qFail := 0
  let mut cApplicable := 0
  let mut cTotal := 0
  for li in [:lines.length] do
    let line := lines[li]!
    let parts := line.splitOn "\t"
    if parts.length < 3 then
      qFail := qFail + 1
      continue
    let qidx := parts[0]!.trim
    let termStr := parts[1]!
    let cands := (parts[2]!.splitOn ",").filterMap (fun c => let t := c.trim; if t.isEmpty then none else some t)
    -- Elaborate/parse the query ONCE per query (amortize over its candidates).
    -- Auto-detect: a field beginning with `{` is a Test B `state` JSON payload
    -- (deserialized with fvars/forall-binders → metavars); otherwise it's a Test A
    -- term string to elaborate through Mathlib.
    let qres ← liftTermElabM do
      let qe? ←
        try
          if termStr.startsWith "{" then
            some <$> TBPS.KernelProbe.parseStateJson termStr
          else
            some <$> TBPS.KernelProbe.elabTermStr env termStr
        catch _ => pure none
      match qe? with
      | none => return ((none : Option Expr), cands.map (fun c => (c, none)))
      | some qe =>
        let results ← cands.mapM (fun c => do
          let nm := TBPS.KernelProbe.parseNameStr c
          let r ← TBPS.KernelProbe.checkApplicable env mode qe nm
          return (c, r))
        return (some qe, results)
    -- The mapM above can't easily mutate counters; recompute counts here.
    match qres with
    | (none, _) =>
      qFail := qFail + 1
      -- emit all candidates as unknown
      for c in cands do
        buf := buf ++ s!"{qidx}\t{c}\t\t{mode}\n"
    | (some _, results) =>
      qOk := qOk + 1
      for (c, r) in results do
        cTotal := cTotal + 1
        match r with
        | some true =>
          cApplicable := cApplicable + 1
          buf := buf ++ s!"{qidx}\t{c}\t1\t{mode}\n"
        | some false =>
          buf := buf ++ s!"{qidx}\t{c}\t0\t{mode}\n"
        | none =>
          buf := buf ++ s!"{qidx}\t{c}\t\t{mode}\n"
    if (li + 1) % 5 == 0 || li + 1 == lines.length then
      IO.FS.Handle.write out buf.toUTF8
      buf := ""
      logInfo m!"[KernelProbe:{mode}] {li+1}/{lines.length} (qOk={qOk}, qFail={qFail}, cTotal={cTotal}, applicable={cApplicable})"
  if !buf.isEmpty then IO.FS.Handle.write out buf.toUTF8
  let pct := Float.ofNat cApplicable / Float.ofNat (max 1 cTotal) * 100.0
  logInfo m!"[KernelProbe:{mode}] done: qOk={qOk} qFail={qFail} cTotal={cTotal} applicable={cApplicable} ({pct}%)"
