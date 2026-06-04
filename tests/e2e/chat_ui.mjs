import { chromium } from "playwright";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "../..");
const BASE = "http://localhost:8011";

async function waitFor(url, timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function main() {
  const server = spawn("python3", ["-m", "uvicorn", "src.main:app", "--port", "8011"], {
    cwd: ROOT,
    stdio: "pipe",
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  server.stderr.on("data", (d) => {
    const msg = d.toString();
    if (msg.includes("Uvicorn running")) process.stdout.write("Server starting...\n");
  });

  console.log("Waiting for server...");
  await waitFor(`${BASE}/api/health`);
  console.log("Server ready");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    await page.goto(BASE);
    await page.waitForSelector("h1");

    // Check UI elements
    const h1 = await page.textContent("h1");
    console.assert(h1 === "Insurance QnA Bot", `h1: ${h1}`);

    const empty = await page.textContent(".empty-state");
    console.assert(empty.includes("Ask a question"), `empty: ${empty}`);

    const disc = await page.textContent(".disclaimer");
    console.assert(disc.includes("AI-generated"), `disclaimer: ${disc}`);

    // Send a question
    await page.fill("input", "What is the premium?");
    await page.click("button[type=submit]");

    // Wait for response to finish streaming
    await page.waitForFunction(
      () => !document.querySelector(".cursor"),
      { timeout: 60000 }
    );

    const answer = await page.textContent(".message.assistant:last-child .bubble");
    console.assert(
      answer.toLowerCase().includes("premium"),
      `answer missing premium: ${answer.slice(0, 80)}`
    );
    console.log("Answer ok:", answer.slice(0, 80));

    // Check sources (M0 has no retrieval, so sources may be empty)
    const sourcesSummary = await page.$("details.sources summary");
    if (sourcesSummary) {
      await sourcesSummary.click();
      const pageNum = await page.textContent(".source-item .page");
      console.assert(pageNum.includes("Page"), `page: ${pageNum}`);
    }

    console.log("PASS");
  } catch (err) {
    console.error("FAIL:", err.message);
    process.exit(1);
  } finally {
    await browser.close();
    server.kill();
  }
}

main();
