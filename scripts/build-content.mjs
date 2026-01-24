import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const contentDir = path.join(rootDir, "content");
const outDir = path.join(rootDir, "src", "generated");

async function readYaml(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return YAML.parse(raw);
}

const chaptersFile = path.join(contentDir, "chapters", "index.yaml");
const achievementsFile = path.join(contentDir, "achievements", "index.yaml");
const problemsDir = path.join(contentDir, "problems");

const chaptersData = await readYaml(chaptersFile);
const achievementsData = await readYaml(achievementsFile);

const problems = [];
const entries = await fs.readdir(problemsDir, { withFileTypes: true });
for (const entry of entries) {
  if (!entry.isFile()) continue;
  if (!entry.name.endsWith(".yaml")) continue;
  const filePath = path.join(problemsDir, entry.name);
  const data = await readYaml(filePath);
  if (data) problems.push(data);
}

problems.sort((a, b) => (a.id || "").localeCompare(b.id || ""));

const output = {
  chapters: chaptersData?.chapters ?? [],
  problems,
  achievements: achievementsData?.achievements ?? []
};

await fs.mkdir(outDir, { recursive: true });
await fs.writeFile(path.join(outDir, "content.json"), JSON.stringify(output, null, 2), "utf8");
