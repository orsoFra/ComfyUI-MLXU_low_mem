<!-- claudoscope-canon: v1 -->
# Project Canon

This repository keeps a canon at `.claude/canon.md`: the settled engineering
decisions that define how and why the code is the way it is. Choices between real
alternatives, constraints, conventions, and hard-won gotchas that are not
recoverable from the code itself. Unlike your per-machine memory, the canon
travels with the repo and is shared by everyone (and every agent) working on it.

## Record format

One `## <Title>` section per record, appended at the end, with a metadata line and
a short body (1-4 sentences, always including the why):

    ## Streaming parser keeps one record type across decode modes
    kind: constraint | date: 2026-07-15 | status: canon
    Lite and full decode share a single raw record type that branches internally.
    Because: the two-type version drifted once and caused silent billing gaps.

- `kind` is one of: `choice`, `constraint`, `convention`, `gotcha`
- `status` is `canon` or `non-canon, superseded by: <newer record title>`

## Reading protocol

1. Before planning, refactoring, reviewing, or answering "why is it like this" or
   "should we" questions: search the canon for terms from the task (subsystem
   names, file names, feature names). If the file is under ~150 lines, read it
   whole. Ignore non-canon records except as history.
2. Records reflect the moment they were written. Verify against the current code
   before relying on one. When code and canon disagree, the code is the truth and
   the record is a candidate for retirement.
3. If the user's request contradicts a canon record, stop and say so, naming the
   record. Let the user decide: follow the canon, or retire the record.
4. When a record shapes your plan, edit, or answer, cite its title.
5. If `.claude/canon.md` does not exist, proceed normally. The canon is opt-in
   per repository.

## Writing protocol

6. When a session settles something that will matter in FUTURE sessions (a choice
   between real alternatives, a new constraint, a non-obvious lesson), offer to
   add it to the canon. Append only after the user agrees. The user can also ask
   directly: "make this canon."
7. Not canon material: task-level choices, TODOs, status updates, anything already
   covered by CLAUDE.md, or anything derivable from the code or git history.
8. Never rewrite or delete existing records. To change one, append a new record
   and flip the old one's status to `non-canon, superseded by: <new title>`.
   History stays intact.
