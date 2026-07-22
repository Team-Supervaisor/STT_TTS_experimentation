"""Vendored subset of the Merchandise AI server, so this repo can run damage
detection without a checkout of merchandise-ai-component next to it.

VENDORED — DO NOT EDIT config.py OR llm.py HERE.

    source:  ~/WWAI/merchandise-ai-component/python-server/app/
    commit:  1f4ad1f  ("Add python server port of the next js app", branch Development)
    files:   config.py, llm.py          <- byte-identical copies of upstream
             damage_detector.py         <- lives HERE, not upstream (see below)
             fixture_config.json        <- generated, see below

Why a package rather than loose files: `damage_detector.py` is written as a
module of the server's `app` package and does `from . import llm`. Giving it any
package to sit in makes that resolve unmodified — the relative import doesn't
care what the package is called. Keeping the file byte-for-byte droppable into
the real `app/` is the point, so nothing here rewrites its imports.

`config.py` loads `.env` from its package's PARENT directory, which here means
the repo root's (gitignored) `.env`. The LLM keys it needs are:

    LLM_PROVIDER, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_VERSION,
    CLAUDE_MODEL

`fixture_config.json` is generated from the app's TypeScript source of truth,
`nextjs-app/lib/merchandise/default-config.ts` — it is a plain object literal, so
it converts directly. Regenerate after any taxonomy change:

    node -e "
    const fs=require('fs');
    let s=fs.readFileSync('nextjs-app/lib/merchandise/default-config.ts','utf8');
    s=s.replace(/^import[^;]*;/m,'')
       .replace(/export const DEFAULT_MERCHANDISE_CONFIG\\s*:\\s*FixtureConfig\\s*=/,'module.exports=');
    fs.writeFileSync('/tmp/cfg.js',s);
    fs.writeFileSync('merch_ai/fixture_config.json',
                     JSON.stringify(require('/tmp/cfg.js'),null,2));"

To re-sync the vendored code, re-copy config.py and llm.py from the source path
above and re-run `multiview_merge_selftest.py`. `damage_detector.py` is the one
file that is authored here — it is the port TARGET for that app, so this repo is
where it changes, and upstream is where it eventually lands.
"""
