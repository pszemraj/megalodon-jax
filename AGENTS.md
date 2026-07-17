# AGENTS.md

High-level:

1. use env as below for running bash/zsh commands
2. if you change code, you run tests wrt that code.
   1. if you change code, `ruff check --fix` and `ruff format --preview` should be run
3. if functions/classes change signature, ensure we have typing and docs updated
4. docs should be updated wrt new features/changes, NOT simply added to.

## protected reproducibility assets

`assets/tokenizer/` is the official tokenizer bundle referenced by the paper and released repository. It is intentionally tracked here for reproducibility and **must remain in this repository**. Do not delete, relocate, ignore, regenerate, replace, or otherwise alter these files without the user's explicit authorization. A large generated-looking diff, lack of runtime imports, or an "orphaned asset" assessment is not justification for removing them.

## git commits

Use atomic commits with conventional commit messages. We will be working in a branch 95%+ of the time, and PRs are merged via squash merge so that the git history is clean.

## ENV INFO

Run commands in the conda env: `conda run --name mega-jax {command}`
(Not needed for agentic tools like `read_file` — only for actual shell execution.)

- Platform: linux-64, CUDA 12.9 available — use the GPU for tests/modeling.
- You can install packages into this env freely.
- Long-running commands are fine; use long timeouts as needed.

---

<!-- SEMBLE_START -->

## Semble — local code search (hybrid keyword + semantic)

`semble` is a search index over <REPO(S)>. It chunks source files, ranks chunks by a blend of BM25 keyword score and semantic-embedding similarity, and returns the top-k matches with `file_path`, line numbers, and the chunk content. It is a **retrieval tool, not an analysis tool**: it finds existing code that resembles a description. It does not reason, judge, summarize, or answer questions. The mental model is "show me the code most like this description" — never "answer this question about the codebase."

NOTE: This was written for Semble 0.5.0. If you have any troubles, please install specifically this version to help debug and move back to something more stable

### Invocation

`semble` is usually installed inside a conda environment rather than globally. Before first use, check this repo's own docs (`CLAUDE.md`, `AGENTS.md`, `README`, contributing/setup notes) for a designated conda env; if one is named, prefix every invocation with it:

```bash
conda run -n <env-name> semble <args>
```

If no env is designated, try `semble --help` bare. Never invoke it via `uvx` — that pulls a separate throwaway installation instead of the one configured here. The examples below omit any prefix; **prepend the `conda run -n <env-name>` prefix if this repo uses one.**

If semble is not installed at all, install it and retry:

```bash
conda run -n <env-name> pip install semble   # into the designated env
# or, with no env in play:
uv tool install semble                       # puts `semble` on $PATH
```

### How to query

A query must describe the *code you expect to find* — its behavior, identifiers, or domain terms — in words that would plausibly appear in or near that code. Queries can be natural language ("retry with exponential backoff") or literal code (a function signature, a distinctive line). Never query with a task, a verdict, or a property the code does not state about itself.

Good — describes target code:

```bash
semble search "retry with exponential backoff"
semble search "parse and validate JWT"
semble search "open a database connection pool"
semble search "def save_pretrained"
```

Bad — a task or judgment, not code that exists under that name:

```bash
semble search "dead code"                # nothing is labeled "dead code"
semble search "duplicate code"           # duplication is a relation you verify, not a string
semble search "security vulnerabilities" # a conclusion you reach by reading code
semble search "what should I refactor"   # not a retrieval query at all
```

The test before running any query: *would these words appear in or near the code I'm looking for?* If not, rewrite the query to describe the behavior instead — or recognize that the task needs reading and reasoning, not retrieval.

### One query, one concept

`search` takes exactly **one** query string. There is no multi-query mode, and the failure is silent: the second positional argument is `path`, so `semble search "retry logic" "auth middleware"` does not search for two things — it tries to index a repo literally named `auth middleware` (or quietly searches the wrong directory if one by that name exists).

Do not compensate by cramming several concepts into one string either. `semble search "auth login session token middleware refresh"` blurs the ranking across all of them and returns mediocre matches for each. When investigating multiple concepts, run one focused search per concept, sequentially:

```bash
semble search "validate session token"
semble search "refresh expired token"
semble search "auth middleware entry point"
```

Each search is ~milliseconds against a cached index; separate calls cost nothing and each returns sharp results.

### Using it for real tasks

Semble surfaces candidates; you do the comparison and the judgment. Example — finding duplicated logic:

1. Do **not** search `"duplicate code"`. Search the behavior you suspect is duplicated: `semble search "normalize and slugify a string"`.
2. Semble returns the top matching chunks across all files, each with `file_path` and line range.
3. Open and compare those chunks yourself to confirm they are actually duplicates.
4. To expand from one known instance, run `find-related` on it (see below).

Same shape for "is this already implemented somewhere," "where else do we do X," and "find handlers like this one": search the behavior, then reason over the hits.

### find-related: from a known location to similar code

`find-related` takes a *location* instead of a query: a `file_path` and 1-indexed `line`, resolves the indexed chunk containing that line, and returns the chunks most similar to it. Use it when you already have one concrete instance in hand and want its siblings — cheaper and more precise than reverse-engineering a text query from the code.

```bash
semble find-related src/text/format.py 88          # chunks similar to the code at that line
semble find-related src/auth.py 42 -k 10           # more results
```

Constraints that matter:

- The file must be **inside the indexed path** (default: cwd). `find-related` looks the chunk up in the index; it does not read arbitrary files. If you get `No chunk found at <file>:<line>`, the file is outside the index root, excluded by ignore rules, or of a content type you didn't index (a YAML file needs `--content config`, a markdown file needs `--content docs`).
- The `line` should land inside the function/block you care about, not on a blank line between definitions.
- Typical flow: `search` → pick the best hit → `find-related` on its `file_path` and start line → compare the returned chunks.

### Checking against a reference codebase or vendored module

Two patterns, depending on where the "other" code lives:

**Both trees under one directory** (vendored module inside this repo, or two checkouts under a common parent): index the common root and `find-related` works across both sides directly.

```bash
# from the common parent containing both my-app/ and vendor/lib/
semble find-related my-app/src/retry.py 30 .       # hits from vendor/lib/ rank alongside my-app/
```

Note that semble honors `.gitignore` and always skips `node_modules/`, `.venv/`, `dist/`, `build/`, etc. If the vendored tree is gitignored, force-include it with a `.sembleignore` entry (gitignore syntax; `!pattern` force-includes).

**Separate repo elsewhere** (a reference codebase at another path, or a git URL): `find-related` cannot span two indexes, so search the *other* repo with a description of the code you have in hand — or use a distinctive signature or line from it as a literal query.

```bash
semble search "token bucket rate limiter" ~/reference/upstream-lib
semble search "def acquire(self, tokens: int)" ~/reference/upstream-lib
semble search "chunk overlap for sliding window" https://github.com/org/repo   # git URLs work; cloned on demand
```

Then open both sides and compare yourself.

### Commands

```bash
semble search "<describe the code>"              # search cwd, top 5 chunks
semble search "<query>" path/to/repo -k 10       # explicit path or git URL, more results
semble search "<query>" --max-snippet-lines 10   # truncate snippets (10 ≈ signature + body head; 0 = locations only)
semble search "<query>" --content docs           # search prose/markdown instead of code
semble search "<query>" --content code config    # multiple content types; also: all
semble find-related <file_path> <line> [path]    # chunks similar to a known location; same -k/--content/--max-snippet-lines flags
semble clear index                               # drop cached indexes if results look stale
```

Defaults: `--content code`, `-k 5`, path = current directory, full chunk content. The index builds on first run, is cached, and rebuilds automatically when files change.

### When to use something else

- Exact string, symbol, or error message → **grep/ripgrep**. Semble ranks by similarity and returns a top-k; it can miss literal hits and never guarantees all of them.
- "Every occurrence of X" (all callers, all usages) → **grep**. Semble is ranked, not exhaustive — never treat its results as a complete enumeration.
- A file whose name or path you already know → read it directly.
- Full context around a hit → open the file at the returned `file_path:line`; don't re-search for the same thing.

Reach for semble when the question is "where is the code that does X" or "what code is like this one." Reach for grep when you already know the exact text or path.

<!-- SEMBLE_END -->
