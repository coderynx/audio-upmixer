#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const webDirectory = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const config = JSON.parse(
  await readFile(path.join(webDirectory, "src-tauri", "tauri.conf.json")),
);

const xml = (value) =>
  String(value).replace(
    /[<>&'"]/g,
    (character) =>
      ({
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "'": "&apos;",
        '"': "&quot;",
      })[character],
  );

const metadata = (executable) => `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleDevelopmentRegion</key><string>English</string>
<key>CFBundleDisplayName</key><string>${xml(config.productName)}</string>
<key>CFBundleExecutable</key><string>${xml(executable)}</string>
<key>CFBundleIdentifier</key><string>${xml(config.identifier)}</string>
<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
<key>CFBundleName</key><string>${xml(config.productName)}</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>${xml(config.version)}</string>
<key>CFBundleVersion</key><string>${xml(config.version)}</string>
<key>CSResourcesFileMapped</key><true/>
<key>LSMinimumSystemVersion</key><string>${xml(config.bundle.macOS.minimumSystemVersion)}</string>
<key>LSRequiresCarbon</key><true/>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>`;

const command = (program, args) =>
  spawnSync(program, args, { encoding: "utf8" });

function appPids(executable) {
  const processes = command("ps", ["-axo", "pid=,command="]).stdout ?? "";
  return processes.split("\n").flatMap((line) => {
    const match = line.trim().match(/^(\d+)\s+(.*)$/);
    return match?.[2].startsWith(executable) ? [Number(match[1])] : [];
  });
}

const pause = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));
const kill = (pid, signal) => {
  try {
    process.kill(pid, signal);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
};

async function stopApp(executable) {
  let pids = appPids(executable);
  if (!pids.length) return;
  for (const pid of pids) kill(pid, "SIGTERM");
  for (let attempt = 0; attempt < 50 && pids.length; attempt += 1) {
    await pause(100);
    pids = appPids(executable);
  }
  for (const pid of pids) kill(pid, "SIGKILL");
}

function digest(file) {
  return spawnSync("shasum", ["-a", "256", file], {
    encoding: "utf8",
  }).stdout.split(" ")[0];
}

async function prepare(source) {
  const executable = path.basename(source);
  const bundle =
    process.env.UPMIXER_DEV_APP ??
    path.join(path.dirname(source), `${config.productName}.app`);
  const bundledExecutable = path.join(bundle, "Contents", "MacOS", executable);
  await stopApp(bundledExecutable);
  await rm(bundle, { recursive: true, force: true });
  await mkdir(path.dirname(bundledExecutable), { recursive: true });
  await cp(source, bundledExecutable);
  await writeFile(
    path.join(bundle, "Contents", "Info.plist"),
    metadata(executable),
  );
  for (const target of Object.values(config.bundle.resources)) {
    const resource = target.replace(/\/$/, "");
    await cp(
      path.join(path.dirname(source), resource),
      path.join(bundle, "Contents", "Resources", resource),
      { recursive: true },
    );
  }
  if (digest(source) !== digest(bundledExecutable))
    throw new Error(
      "development bundle executable does not match Cargo output",
    );
  console.log(`Development bundle: ${bundle}`);
  console.log(
    `Executable SHA-256: ${spawnSync("shasum", ["-a", "256", bundledExecutable], { encoding: "utf8" }).stdout.trim()}`,
  );
  return { bundle, bundledExecutable };
}

async function main() {
  const prepareOnly = process.argv[2] === "--prepare";
  const supplied = process.argv[prepareOnly ? 3 : 2];
  if (!supplied) throw new Error("Cargo did not provide an executable");
  const source = path.resolve(supplied);
  const { bundle, bundledExecutable } = await prepare(source);
  if (prepareOnly) return;

  const open = spawn(
    "open",
    ["-n", "-W", bundle, "--args", ...process.argv.slice(3)],
    { stdio: "inherit" },
  );
  let reported = false;
  const report = setInterval(() => {
    const pid = appPids(bundledExecutable)[0];
    if (!pid || reported) return;
    const identity = command("lsappinfo", [
      "info",
      "-only",
      "CFBundleIdentifier,LSBundlePath,LSLaunchedByLaunchServices,pid",
      String(pid),
    ]).stdout.trim();
    if (/LSLaunchedByLaunchServices"?\s*=\s*true/.test(identity)) {
      reported = true;
      console.log(`Launch Services launch:\n${identity}`);
    }
  }, 100);
  const cleanup = async () => {
    clearInterval(report);
    open.kill("SIGTERM");
    await stopApp(bundledExecutable);
  };
  for (const signal of ["SIGINT", "SIGTERM"])
    process.once(signal, () => cleanup().finally(() => process.exit(130)));
  await new Promise((resolve, reject) =>
    open.once("error", reject).once("exit", resolve),
  );
  clearInterval(report);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
