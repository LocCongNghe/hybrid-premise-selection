import Mathlib
import Qq

open Lean Elab Term Meta Json

namespace TBPS

inductive ExtractedExpr where
  | bvar (deBruijnIndex : Nat)
  | fvar (fvarId : String)
  | mvar (mvarId : String)
  | sort (u : String)
  | const (declName : String) (us : List String)
  | app (fn : ExtractedExpr) (arg : ExtractedExpr)
  | lam (binderName : String) (binderType : ExtractedExpr) (body : ExtractedExpr)
      (binderInfo : String)
  | forallE (binderName : String) (binderType : ExtractedExpr) (body : ExtractedExpr)
      (binderInfo : String)
  | letE (declName : String) (type : ExtractedExpr) (value : ExtractedExpr)
      (body : ExtractedExpr) (nonDep : Bool)
  | lit (literal : String)
  | mdata (data : String) (expr : ExtractedExpr)
  | proj (typeName : String) (idx : Nat) (struct : ExtractedExpr)
  deriving ToJson, FromJson, Repr, Inhabited

structure ExtractFrame where
  node : Expr
  childResults : Array ExtractedExpr
  remaining : List Expr
  deriving Inhabited

def extractedChildren (e : Expr) : List Expr :=
  match e with
  | .app f a => [f, a]
  | .lam _ t b _ => [t, b]
  | .forallE _ t b _ => [t, b]
  | .letE _ t v b _ => [t, v, b]
  | .mdata _ e => [e]
  | .proj _ _ s => [s]
  | _ => []

def reconstructExtracted (e : Expr) (children : List ExtractedExpr) : ExtractedExpr :=
  match e with
  | .bvar idx => .bvar idx
  | .fvar id => .fvar ((repr id).pretty)
  | .mvar id => .mvar ((repr id).pretty)
  | .sort level => .sort ((repr level).pretty)
  | .const name levels => .const name.toString (levels.map fun level => (repr level).pretty)
  | .app _ _ => match children with | [f, a] => .app f a | _ => panic! "app children"
  | .lam name _ _ info => match children with
      | [type, body] => .lam name.toString type body ((repr info).pretty)
      | _ => panic! "lambda children"
  | .forallE name _ _ info => match children with
      | [type, body] => .forallE name.toString type body ((repr info).pretty)
      | _ => panic! "forall children"
  | .letE name _ _ _ nonDep => match children with
      | [type, value, body] => .letE name.toString type value body nonDep
      | _ => panic! "let children"
  | .lit literal => .lit ((repr literal).pretty)
  | .mdata data _ => match children with
      | [expr] => .mdata ((repr data).pretty) expr
      | _ => panic! "metadata children"
  | .proj typeName idx _ => match children with
      | [struct] => .proj typeName.toString idx struct
      | _ => panic! "projection children"

partial def exprToExtracted (e : Expr) (maxDepth : Nat := 1000) : MetaM ExtractedExpr := do
  let mut stack : List ExtractFrame := [
    { node := e, childResults := #[], remaining := extractedChildren e }
  ]
  while !stack.isEmpty do
    if stack.length > maxDepth then
      throwError "Iteration depth exceeded threshold"
    let top := stack.head!
    match top.remaining with
    | child :: rest =>
        stack := { top with remaining := rest } :: stack.tail!
        stack := { node := child, childResults := #[], remaining := extractedChildren child } :: stack
    | [] =>
        let result := reconstructExtracted top.node top.childResults.toList
        stack := stack.tail!
        if stack.isEmpty then
          return result
        let parent := stack.head!
        stack := { parent with childResults := parent.childResults.push result } :: stack.tail!
  unreachable!

def parseExtractedTerm (input : String) : TermElabM Expr := do
  let environment ← getEnv
  match Lean.Parser.runParserCategory environment `term input with
  | .ok stx => elabTerm stx none
  | .error error => throwError s!"parser error: {error}"

def extractTermJson (input : String) : TermElabM Json := do
  let expression ← instantiateMVars (← parseExtractedTerm input)
  let extracted ← exprToExtracted expression 1000
  pure <| Json.mkObj [
    ("input_str", Json.str input),
    ("expr_dbg", Json.str expression.dbgToString),
    ("your_expr", toJson extracted)
  ]

syntax (name := tbpsParseAndWrite) "tbps_parse_and_write " str str : command

elab_rules : command
  | `(tbps_parse_and_write $inputPath:str $outputPath:str) => do
      let input ← IO.FS.readFile inputPath.getString
      let json ← Lean.Elab.Command.runTermElabM fun _ => extractTermJson input.trim
      IO.FS.writeFile outputPath.getString json.compress

end TBPS
