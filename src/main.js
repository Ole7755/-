import "./style.css";
import content from "./generated/content.json";

const app = document.querySelector("#app");
const STORAGE_KEY = "vimdojo-progress-v1";

let progress = loadProgress();
render();

function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { completed: [] };
    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.completed)) return { completed: [] };
    return { completed: data.completed };
  } catch {
    return { completed: [] };
  }
}

function saveProgress(next) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
}

function exportProgress() {
  const blob = new Blob([JSON.stringify(progress, null, 2)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "vimdojo-progress.json";
  a.click();
  URL.revokeObjectURL(url);
}

async function importProgress(file) {
  const text = await file.text();
  const data = JSON.parse(text);
  if (!data || !Array.isArray(data.completed)) {
    throw new Error("格式不正确");
  }
  progress = { completed: data.completed };
  saveProgress(progress);
  render();
}

function render() {
  const chaptersHtml = content.chapters
    .slice()
    .sort((a, b) => a.order - b.order)
    .map(
      (chapter) =>
        `<button type="button" data-chapter="${chapter.id}">${chapter.title}</button>`
    )
    .join("");

  const problemsHtml = content.problems.length
    ? content.problems
        .map(
          (problem) =>
            `<button type="button" data-problem="${problem.id}">${problem.title}</button>`
        )
        .join("")
    : `<div class="muted">暂无题目，请在 content/problems 目录添加。</div>`;

  app.innerHTML = `
    <header>
      <h1>VimDojo</h1>
      <div class="actions">
        <button type="button" id="exportBtn">导出进度</button>
        <button type="button" id="importBtn">导入进度</button>
        <button type="button" id="playgroundBtn">自由练习区</button>
      </div>
    </header>
    <main>
      <aside>
        <div class="section-title">章节</div>
        <div class="list">${chaptersHtml}</div>
        <div class="section-title">进度</div>
        <div class="muted">已完成 ${progress.completed.length} / ${content.problems.length}</div>
      </aside>
      <section class="content">
        <div class="card">
          <div class="section-title">题库</div>
          <div class="list">${problemsHtml}</div>
        </div>
        <div class="card">
          <div class="section-title">题目区域（占位）</div>
          <div class="muted">题面、提示、提交按钮与差异查看将在这里显示。</div>
        </div>
        <div class="card">
          <div class="section-title">编辑器（占位）</div>
          <div class="editor">CodeMirror + Vim 键位将在此接入。</div>
        </div>
      </section>
    </main>
    <input type="file" id="importFile" accept="application/json" style="display:none" />
  `;

  const exportBtn = document.querySelector("#exportBtn");
  const importBtn = document.querySelector("#importBtn");
  const importFile = document.querySelector("#importFile");

  exportBtn.addEventListener("click", exportProgress);
  importBtn.addEventListener("click", () => importFile.click());
  importFile.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      await importProgress(file);
      alert("导入成功");
    } catch (error) {
      alert(`导入失败：${error.message || "格式错误"}`);
    } finally {
      importFile.value = "";
    }
  });
}
