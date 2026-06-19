import { chmod, cp, mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { deflateSync } from "node:zlib";

const appName = "QuantDesk";
const bundleId = "com.quantdesk.local";
const projectRoot = process.cwd();
const deploymentTarget = "13.0";
const swiftArch = process.arch === "x64" ? "x86_64" : "arm64";
const swiftTarget = `${swiftArch}-apple-macosx${deploymentTarget}`;
const distDir = join(projectRoot, "dist");
const bundleDir = join(distDir, `${appName}.app`);
const contentsDir = join(bundleDir, "Contents");
const macosDir = join(contentsDir, "MacOS");
const resourcesDir = join(contentsDir, "Resources");
const bundledAppDir = join(resourcesDir, "app");
const assetsDir = join(projectRoot, "assets");
const iconPng = join(assetsDir, "quantdesk-icon.png");
const iconsetDir = join(distDir, "QuantDesk.iconset");
const iconIcns = join(resourcesDir, "QuantDesk.icns");

async function main() {
  assertSourceFile("server.py");
  assertSourceFile("static/index.html");
  assertSourceFile("static/echarts.min.js");
  assertSourceFile("macos/QuantDeskApp.swift");

  await rm(bundleDir, { recursive: true, force: true });
  await mkdir(macosDir, { recursive: true });
  await mkdir(bundledAppDir, { recursive: true });
  await mkdir(assetsDir, { recursive: true });

  await createIconAssets();
  await cp(join(projectRoot, "server.py"), join(bundledAppDir, "server.py"));
  await cp(join(projectRoot, "static"), join(bundledAppDir, "static"), {
    recursive: true
  });

  if (existsSync(join(projectRoot, "sina_data.py"))) {
    await cp(join(projectRoot, "sina_data.py"), join(bundledAppDir, "sina_data.py"));
  }

  await writeFile(join(contentsDir, "Info.plist"), infoPlist());
  await writeFile(join(contentsDir, "PkgInfo"), "APPL????");
  await run("/usr/bin/swiftc", [
    "-target",
    swiftTarget,
    "-parse-as-library",
    "macos/QuantDeskApp.swift",
    "-o",
    join(macosDir, appName),
    "-framework",
    "Cocoa",
    "-framework",
    "WebKit"
  ]);
  await chmod(join(macosDir, appName), 0o755);

  console.log(`Built ${bundleDir}`);
  console.log(`Run: open "${bundleDir}"`);
}

function assertSourceFile(relativePath: string) {
  if (!existsSync(join(projectRoot, relativePath))) {
    throw new Error(`Missing required file: ${relativePath}`);
  }
}

function infoPlist() {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>zh_CN</string>
  <key>CFBundleDisplayName</key>
  <string>${appName}</string>
  <key>CFBundleExecutable</key>
  <string>${appName}</string>
  <key>CFBundleIconFile</key>
  <string>QuantDesk</string>
  <key>CFBundleIdentifier</key>
  <string>${bundleId}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${appName}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleSignature</key>
  <string>????</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0.0</string>
  <key>CFBundleSupportedPlatforms</key>
  <array>
    <string>MacOSX</string>
  </array>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
  <key>LSUIElement</key>
  <false/>
  <key>NSAppTransportSecurity</key>
  <dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
    <key>NSAllowsArbitraryLoadsInWebContent</key>
    <true/>
  </dict>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
</dict>
</plist>
`;
}

async function createIconAssets() {
  await writeFile(iconPng, renderIconPng(1024));
  await rm(iconsetDir, { recursive: true, force: true });
  await mkdir(iconsetDir, { recursive: true });

  const sizes = [
    ["icon_16x16.png", 16],
    ["icon_16x16@2x.png", 32],
    ["icon_32x32.png", 32],
    ["icon_32x32@2x.png", 64],
    ["icon_128x128.png", 128],
    ["icon_128x128@2x.png", 256],
    ["icon_256x256.png", 256],
    ["icon_256x256@2x.png", 512],
    ["icon_512x512.png", 512],
    ["icon_512x512@2x.png", 1024]
  ] as const;

  for (const [name, size] of sizes) {
    await run("/usr/bin/sips", ["-z", String(size), String(size), iconPng, "--out", join(iconsetDir, name)]);
  }

  await run("/usr/bin/iconutil", ["-c", "icns", iconsetDir, "-o", iconIcns]);
}

async function run(command: string, args: string[]) {
  const proc = Bun.spawn([command, ...args], {
    cwd: projectRoot,
    env: {
      ...process.env,
      MACOSX_DEPLOYMENT_TARGET: deploymentTarget
    },
    stdout: "pipe",
    stderr: "pipe"
  });
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`${command} failed: ${stderr}`);
  }
}

function renderIconPng(size: number) {
  const rgba = new Uint8Array(size * size * 4);
  const icon = new Bitmap(rgba, size, size);

  const corner = 220;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const alpha = roundedRectCoverage(x + 0.5, y + 0.5, 34, 34, size - 68, size - 68, corner);
      if (alpha <= 0) {
        continue;
      }
      const gx = x / size;
      const gy = y / size;
      const glow = Math.max(0, 1 - distance(x, y, 740, 250) / 760);
      const tealGlow = Math.max(0, 1 - distance(x, y, 260, 730) / 620);
      icon.blendPixel(x, y, [
        Math.round(7 + 16 * gx + 15 * glow),
        Math.round(14 + 22 * gy + 42 * tealGlow),
        Math.round(24 + 34 * gx + 54 * glow)
      ], alpha);
    }
  }

  icon.roundedRect(86, 86, 852, 852, 156, [22, 36, 54], 0.62);
  icon.roundedRect(126, 128, 772, 560, 54, [9, 16, 25], 0.52);

  for (let i = 0; i <= 8; i += 1) {
    const x = 146 + i * 92;
    icon.line(x, 148, x, 672, [88, 124, 162], 2, 0.20);
  }
  for (let i = 0; i <= 5; i += 1) {
    const y = 168 + i * 92;
    icon.line(138, y, 888, y, [88, 124, 162], 2, 0.20);
  }

  icon.line(150, 570, 278, 480, [32, 214, 199], 12, 0.85);
  icon.line(278, 480, 410, 520, [32, 214, 199], 12, 0.85);
  icon.line(410, 520, 548, 350, [32, 214, 199], 12, 0.85);
  icon.line(548, 350, 702, 392, [32, 214, 199], 12, 0.85);
  icon.line(702, 392, 854, 230, [32, 214, 199], 12, 0.85);

  const candles = [
    [198, 270, 585, 526, 96, [53, 208, 127]],
    [314, 210, 610, 302, 132, [255, 95, 109]],
    [432, 250, 580, 418, 112, [53, 208, 127]],
    [552, 176, 525, 232, 146, [255, 95, 109]],
    [676, 190, 542, 326, 126, [53, 208, 127]],
    [800, 126, 498, 184, 136, [53, 208, 127]]
  ] as const;

  for (const [x, high, low, bodyY, bodyH, color] of candles) {
    icon.line(x, high, x, low, color, 7, 0.95);
    icon.roundedRect(x - 28, bodyY, 56, bodyH, 12, color, 0.96);
  }

  icon.roundedRect(142, 728, 740, 126, 36, [7, 13, 20], 0.72);
  icon.line(180, 816, 482, 760, [77, 163, 255], 8, 0.90);
  icon.line(482, 760, 846, 792, [247, 185, 85], 8, 0.88);
  icon.drawBlockText("QD", 226, 744, 18, [236, 248, 255], 0.95);
  icon.drawBlockText("AI", 650, 744, 18, [32, 214, 199], 0.94);

  icon.roundedRect(34, 34, size - 68, size - 68, corner, [255, 255, 255], 0.08, true);
  icon.roundedRect(70, 70, size - 140, size - 140, 178, [32, 214, 199], 0.10, true);

  return encodePng(size, size, rgba);
}

class Bitmap {
  constructor(private rgba: Uint8Array, private width: number, private height: number) {}

  blendPixel(x: number, y: number, color: readonly number[], alpha: number) {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height || alpha <= 0) {
      return;
    }
    const i = (Math.floor(y) * this.width + Math.floor(x)) * 4;
    const srcA = Math.max(0, Math.min(1, alpha));
    const dstA = this.rgba[i + 3] / 255;
    const outA = srcA + dstA * (1 - srcA);
    if (outA <= 0) {
      return;
    }
    this.rgba[i] = Math.round((color[0] * srcA + this.rgba[i] * dstA * (1 - srcA)) / outA);
    this.rgba[i + 1] = Math.round((color[1] * srcA + this.rgba[i + 1] * dstA * (1 - srcA)) / outA);
    this.rgba[i + 2] = Math.round((color[2] * srcA + this.rgba[i + 2] * dstA * (1 - srcA)) / outA);
    this.rgba[i + 3] = Math.round(outA * 255);
  }

  roundedRect(x: number, y: number, w: number, h: number, r: number, color: readonly number[], alpha: number, stroke = false) {
    const minX = Math.max(0, Math.floor(x - 4));
    const minY = Math.max(0, Math.floor(y - 4));
    const maxX = Math.min(this.width, Math.ceil(x + w + 4));
    const maxY = Math.min(this.height, Math.ceil(y + h + 4));
    for (let py = minY; py < maxY; py += 1) {
      for (let px = minX; px < maxX; px += 1) {
        const coverage = roundedRectCoverage(px + 0.5, py + 0.5, x, y, w, h, r);
        if (stroke) {
          const inner = roundedRectCoverage(px + 0.5, py + 0.5, x + 6, y + 6, w - 12, h - 12, Math.max(0, r - 6));
          this.blendPixel(px, py, color, Math.max(0, coverage - inner) * alpha);
        } else {
          this.blendPixel(px, py, color, coverage * alpha);
        }
      }
    }
  }

  line(x1: number, y1: number, x2: number, y2: number, color: readonly number[], width: number, alpha: number) {
    const pad = width + 3;
    const minX = Math.max(0, Math.floor(Math.min(x1, x2) - pad));
    const minY = Math.max(0, Math.floor(Math.min(y1, y2) - pad));
    const maxX = Math.min(this.width, Math.ceil(Math.max(x1, x2) + pad));
    const maxY = Math.min(this.height, Math.ceil(Math.max(y1, y2) + pad));
    for (let y = minY; y < maxY; y += 1) {
      for (let x = minX; x < maxX; x += 1) {
        const d = distanceToSegment(x + 0.5, y + 0.5, x1, y1, x2, y2);
        const coverage = Math.max(0, Math.min(1, width / 2 + 0.8 - d));
        this.blendPixel(x, y, color, coverage * alpha);
      }
    }
  }

  drawBlockText(text: string, x: number, y: number, unit: number, color: readonly number[], alpha: number) {
    const glyphs: Record<string, string[]> = {
      Q: ["11110", "10010", "10010", "10010", "10110", "10010", "11101"],
      D: ["11110", "10010", "10010", "10010", "10010", "10010", "11110"],
      A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
      I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"]
    };
    let cursor = x;
    for (const char of text) {
      const rows = glyphs[char] || [];
      rows.forEach((row, rowIndex) => {
        [...row].forEach((cell, colIndex) => {
          if (cell === "1") {
            this.roundedRect(cursor + colIndex * unit, y + rowIndex * unit, unit * 0.78, unit * 0.78, unit * 0.18, color, alpha);
          }
        });
      });
      cursor += unit * 6;
    }
  }
}

function roundedRectCoverage(px: number, py: number, x: number, y: number, w: number, h: number, r: number) {
  const cx = Math.max(x + r, Math.min(px, x + w - r));
  const cy = Math.max(y + r, Math.min(py, y + h - r));
  const d = distance(px, py, cx, cy) - r;
  if (px >= x + r && px <= x + w - r && py >= y && py <= y + h) {
    return 1;
  }
  if (px >= x && px <= x + w && py >= y + r && py <= y + h - r) {
    return 1;
  }
  return Math.max(0, Math.min(1, 1 - d));
}

function distance(x1: number, y1: number, x2: number, y2: number) {
  return Math.hypot(x1 - x2, y1 - y2);
}

function distanceToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lengthSq));
  return distance(px, py, x1 + t * dx, y1 + t * dy);
}

function encodePng(width: number, height: number, rgba: Uint8Array) {
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * (width * 4 + 1);
    raw[rowStart] = 0;
    Buffer.from(rgba.buffer, y * width * 4, width * 4).copy(raw, rowStart + 1);
  }

  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", Buffer.concat([
      uint32(width),
      uint32(height),
      Buffer.from([8, 6, 0, 0, 0])
    ])),
    pngChunk("IDAT", deflateSync(raw, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0))
  ]);
}

function pngChunk(type: string, data: Buffer) {
  const typeBuffer = Buffer.from(type);
  const crcInput = Buffer.concat([typeBuffer, data]);
  return Buffer.concat([
    uint32(data.length),
    typeBuffer,
    data,
    uint32(crc32(crcInput))
  ]);
}

function uint32(value: number) {
  const buffer = Buffer.alloc(4);
  buffer.writeUInt32BE(value >>> 0);
  return buffer;
}

function crc32(buffer: Buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let i = 0; i < 8; i += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
