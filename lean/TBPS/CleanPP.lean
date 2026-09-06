import Mathlib
import Qq

open Lean Elab Command Meta

/-!
# Clean pretty-print of Mathlib declaration statements (I3 dense-space fix)

The dense retriever (`kaiyuy/leandojo-lean4-retriever-byt5-small`) was trained on **clean,
source-like** Lean statement text (`∀ (n m : ℕ), n + m = m + n`), but the corpus column
`mathlib_filtered.statement_str` is the **fully elaborated** pretty-print
(`forall (n m : Nat), Eq.{1} Nat (HAdd.hAdd.{0,0,0} ... (instHAdd.{0} Nat instAddNat) n m) ...`).
Embedding docs from the elaborated form while embedding queries from the raw `q(...)` form put
query and document in **different spaces** → dense cosine was noise (correct-target 0.058 vs a
wrong doc 0.516).

This module regenerates the **clean** statement text for every corpus premise by re-pretty-printing
the elaborated `Expr` with Lean's **default** printer (`pp.all := false`): notation on (`+`, `=`,
`∀`, `ℕ`), universe levels and typeclass instances hidden. Verified across all declaration kinds
(thm/def/induct/ctor/rec/projection/instance) on `Nat.add_comm`, `Nat.gcd`, `List`, `List.cons`,
`Eq.ndrec`, `instAddNat`, etc.

Usage (run from the `lean/` dir so `lake env` sets LEAN_PATH to the Mathlib oleans):

  # Docs: clean-pp the statement (info.type) of every premise name in NAMES_FILE.
  lake env lean --run TBPS/CleanPP.lean docs NAMES_FILE OUT_FILE

  # Queries: clean-pp raw Test A query strings (the `q(...)` lines) in QUERIES_FILE.
  lake env lean --run TBPS/CleanPP.lean queries QUERIES_FILE OUT_FILE

Output is TSV: `name\tclean_text` (docs) or `idx\tclean_text` (queries), one record per line.
Names not found / queries that fail to parse are still emitted with their raw input so no record
is lost; failures are logged to stderr.
-/

namespace TBPS.CleanPP

/-- Strip the Qq `q(...)` quotation wrapper stored in the official Test A text file.
Mirrors `lean_extractor.normalize_test_a_expression`. -/
def stripQqWrapper (s : String) : String :=
  let t := s.trim
  if t.startsWith "q(" && t.endsWith ")" then
    (t.drop 2).dropRight 1 |>.trim
  else t

/-- Per-declaration heartbeat budget for pretty-printing. A handful of pathological Mathlib
declarations (e.g. heavy CategoryTheory iso hom-apps) make `ppExpr` run away and either exhaust
memory or hang the whole 217k batch — even with `maxHeartbeats 0` (infinite) set at the driver
level. Capping each declaration to a finite quota turns such a declaration into a caught timeout
→ `none` → empty TSV line, instead of crashing/stalling the entire batch. 400000 ≈ 2× the default. -/
def ppHeartbeatBudget := 400000

/-- Clean pretty-print of an elaborated `Expr` using Lean's default (non-`pp.all`) printer.
Returns `none` if pretty-printing throws or exceeds the per-declaration heartbeat budget
(caller falls back to raw/empty text). -/
def cleanPPExpr (e : Expr) : TermElabM (Option String) := do
  try
    -- Run ppExpr under a FINITE per-declaration heartbeat cap, regardless of the global
    -- maxHeartbeats setting. `withOptions` scopes the option to just this call. A declaration
    -- that blows the quota raises a `(deterministic) timeout` Exception, caught below.
    let fmt ← withOptions (fun o => o.set `maxHeartbeats ppHeartbeatBudget) do
      PrettyPrinter.ppExpr e
    return some fmt.pretty
  catch _ =>
    return none

end TBPS.CleanPP

/-- Parse a dotted Lean name string like ``Nat.add_comm`` into a `Name`. -/
def parseNameStr (s : String) : Name :=
  let parts := s.split (· == '.')
  parts.foldl (fun acc p => if p.isEmpty then acc else Name.str acc p) Name.anonymous

/-- Core logic for docs mode: clean-pp `info.type` of every premise name in `namesFile`,
write `name\tclean_text` per line to `outFile`. Run as `elab_rules` so it executes during
elaboration in the Mathlib-loaded environment provided by `lake env lean`. -/
elab (name := tbpsCleanPPDocs) "tbps_clean_pp_docs " namesFile:str outFile:str : command => do
  let names ← IO.FS.readFile namesFile.getString
  let entries := names.split (· == '\n') |>.filterMap (fun l => let t := l.trim; if t.isEmpty then none else some t)
  logInfo m!"[CleanPP:docs] {entries.length} names from {namesFile.getString} → {outFile.getString}"
  let env ← getEnv
  let out ← IO.FS.Handle.mk outFile.getString .write
  -- Process in BATCHES of `batchSize`, each batch wrapped in its OWN `liftTermElabM`.
  -- Rationale: a single `liftTermElabM` over all 217k iterations threads the Meta state
  -- (mvar table, lcatalog, metavar context) through every `ppExpr` call, and on complex
  -- CategoryTheory declarations that state accumulates without being released — RSS climbs
  -- until the WSL2 OOM killer terminates the process mid-write (observed: crash at ~30k with
  -- a truncated final line and no error, all buffered `logInfo` lost). Per-name `liftTermElabM`
  -- avoids the leak but is ~10× slower (setup cost). Batching at 1000 names/batch gives a
  -- fresh Meta state every 1000 names (bounding memory) while amortizing the setup cost
  -- (~217 setups total instead of 217k). File writes happen via liftIO outside the batch
  -- monad; records are buffered into a String and flushed every 5000 names.
  let batchSize := 1000
  let mut buf : String := ""
  let mut emitted := 0
  let mut notFound := 0
  let mut ppFail := 0
  let mut i := 0
  while i < entries.length do
    let endIdx := min (i + batchSize) entries.length
    let batch ← liftTermElabM do
      let mut b : String := ""
      let mut em := 0
      let mut nf := 0
      let mut pf := 0
      for j in List.range' i (endIdx - i) do
        let nmStr := entries[j]!
        let nm := parseNameStr nmStr
        match env.find? nm with
        | none =>
          nf := nf + 1
          b := b ++ s!"{nmStr}\t\n"
        | some info =>
          let t : Expr := info.type
          let res ← TBPS.CleanPP.cleanPPExpr t
          match res with
          | some clean =>
            let esc := clean.replace "\n" " "
            b := b ++ s!"{nmStr}\t{esc}\n"
            em := em + 1
          | none =>
            pf := pf + 1
            b := b ++ s!"{nmStr}\t\n"
      return (b, em, nf, pf)
    let (b, em, nf, pf) := batch
    buf := buf ++ b
    emitted := emitted + em
    notFound := notFound + nf
    ppFail := ppFail + pf
    i := endIdx
    if i % 5000 == 0 || i == entries.length then
      IO.FS.Handle.write out buf.toUTF8
      buf := ""
      logInfo m!"[CleanPP:docs] {i}/{entries.length} (emitted={emitted}, notFound={notFound}, ppFail={ppFail})"
  if !buf.isEmpty then
    IO.FS.Handle.write out buf.toUTF8
  logInfo m!"[CleanPP:docs] done: emitted={emitted} notFound={notFound} ppFail={ppFail} (total={entries.length})"

/-- Core logic for queries mode: clean-pp each raw Test A query string (strip `q(...)`,
parse + elaborate + instantiateMVars + clean-pp), write `idx\tclean_text` per line to `outFile`.
Mirrors `ExtractExpr.extractTermJson` but outputs the clean pretty-print instead of the JSON tree. -/
elab (name := tbpsCleanPPQueries) "tbps_clean_pp_queries " queriesFile:str outFile:str : command => do
  let qs ← IO.FS.readFile queriesFile.getString
  let entries := qs.split (· == '\n') |>.filterMap (fun l => let t := l.trim; if t.isEmpty then none else some t)
  logInfo m!"[CleanPP:queries] {entries.length} queries from {queriesFile.getString} → {outFile.getString}"
  let env ← getEnv
  let mut out ← IO.FS.Handle.mk outFile.getString .write
  let mut ok := 0
  let mut fail := 0
  for i in [:entries.length] do
    let raw := entries[i]!
    let stripped := TBPS.CleanPP.stripQqWrapper raw
    -- res : Option String — `some clean` on success, `none` on any failure (fall back to raw).
    let res ← liftTermElabM do
      match Lean.Parser.runParserCategory env `term stripped with
      | .error err =>
        logInfo m!"[CleanPP:queries] {i}: parse error: {err}"
        return none
      | .ok stx =>
        try
          let e ← Elab.Term.elabTerm stx none
          let e ← instantiateMVars e
          TBPS.CleanPP.cleanPPExpr e
        catch ex =>
          logInfo m!"[CleanPP:queries] {i}: elab error: {ex.toMessageData}"
          return none
    match res with
    | some clean =>
      let esc := clean.replace "\n" " "
      out.putStr s!"{i}\t{esc}\n"
      ok := ok + 1
    | none =>
      -- clean-pp/elab failed; still record the stripped raw text so the query is not lost.
      out.putStr s!"{i}\t{stripped}\n"
      fail := fail + 1
    if (i + 1) % 20 == 0 then
      out.flush
      logInfo m!"[CleanPP:queries] {i+1}/{entries.length} (ok={ok}, fail={fail})"
  out.flush
  logInfo m!"[CleanPP:queries] done: ok={ok} fail={fail} (total={entries.length})"
