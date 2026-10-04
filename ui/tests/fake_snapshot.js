// Node helper for tests/test_web.py: runs the UI's demo world headless and prints JSON.
//   node ui/tests/fake_snapshot.js snapshot 20.5   -> one snapshot at script time 20.5 s
//   node ui/tests/fake_snapshot.js flow            -> the full PHASE / SPEAK / HAPTIC history of one demo loop
require("../js/sources.js");
const { createFake } = globalThis.Illumin.sources;
const [mode, arg] = process.argv.slice(2);
if (mode === "flow") {
  const f = createFake({ at: 48 });
  process.stdout.write(JSON.stringify(f.history));
} else {
  const f = createFake({ at: parseFloat(arg) || 0 });
  process.stdout.write(JSON.stringify(f.snapshot()));
}
