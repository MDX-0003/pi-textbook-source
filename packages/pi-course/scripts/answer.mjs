import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const repoRoot = path.resolve(packageRoot, "..", "..");
const requested = process.argv[2]?.padStart(2, "0");

if (!requested || !/^\d{2}$/.test(requested)) {
  process.stderr.write(
    "用法：npm run answer -w @pi/course -- <00..14> [输出目录]\n",
  );
  process.exit(1);
}

// 找到 target commit（与 checkpoint.mjs 同样的查找逻辑）
const rows = execFileSync(
  "git",
  ["log", "--reverse", "--format=%H%x09%P%x09%s", "--", "packages/pi-course"],
  { cwd: repoRoot, encoding: "utf8" },
)
  .trim()
  .split("\n")
  .map((line) => {
    const [commit, parents, subject] = line.split("\t");
    return { commit, parent: parents.split(" ")[0], subject };
  });
const row = rows.find(({ subject }) =>
  subject.startsWith(`course(${requested}):`),
);

if (!row) {
  process.stderr.write(`找不到 checkpoint ${requested}\n`);
  process.exit(1);
}

const output = process.argv[3]
  ? path.resolve(repoRoot, process.argv[3])
  : path.resolve(repoRoot, "..", `pi-answer-${requested}`);

if (existsSync(output)) {
  process.stderr.write(`输出目录已存在：${output}\n`);
  process.exit(1);
}

// 用 git worktree 提取 target commit 的完整文件（避免 tar 在 Windows 上的路径问题）
const worktree = await mkdtemp(path.join(os.tmpdir(), "pi-answer-wt-"));
let createdOutput = false;

try {
  execFileSync(
    "git",
    ["worktree", "add", "--detach", worktree, row.commit],
    { cwd: repoRoot, stdio: "pipe" },
  );

  await mkdir(output, { recursive: false });
  createdOutput = true;

  // 只复制需要的文件
  await cp(path.join(worktree, "package.json"), path.join(output, "package.json"));
  await cp(path.join(worktree, "package-lock.json"), path.join(output, "package-lock.json"));
  await cp(path.join(worktree, "packages"), path.join(output, "packages"), { recursive: true });

  // 去掉 prepare 脚本避免安装时出问题
  const rootPackagePath = path.join(output, "package.json");
  const rootPackageDocument = JSON.parse(
    await readFile(rootPackagePath, "utf8"),
  );
  if (rootPackageDocument.scripts) {
    delete rootPackageDocument.scripts.prepare;
  }
  await writeFile(
    rootPackagePath,
    `${JSON.stringify(rootPackageDocument, null, "\t")}\n`,
  );

  // 找聚焦测试
  const testFiles = execFileSync(
    "git",
    ["ls-tree", "-r", "--name-only", row.commit, "packages/pi-course/test"],
    { cwd: repoRoot, encoding: "utf8" },
  )
    .trim()
    .split("\n");
  const focusedTest = testFiles.find((file) =>
    file.startsWith(`packages/pi-course/test/${requested}-`),
  );
  if (!focusedTest) {
    throw new Error(`checkpoint ${requested} 缺少聚焦测试`);
  }

  const packageDocument = JSON.parse(
    await readFile(
      path.join(output, "packages/pi-course/package.json"),
      "utf8",
    ),
  );
  const packageName = packageDocument.name;

  const guide = `# Chapter ${requested} 答案

模式：答案（target commit 完整源码）

- target（本章完成态）：\`${row.commit}\`
- parent（本章起点）：\`${row.parent}\`
- 聚焦测试：\`${focusedTest}\`

这个目录包含 Chapter ${requested} 的**最终完成代码**。所有测试应当直接通过。

第一次运行：

\`\`\`bash
npm install
npm run build -w ${packageName}
node --test packages/pi-course/dist/test/${requested}-*.test.js
\`\`\`

对比自己实现与答案的差异：

\`\`\`bash
git diff ${row.parent} ${row.commit} -- packages/pi-course
\`\`\`
`;
  await writeFile(path.join(output, "LEARNING.md"), guide);

  process.stdout.write(
    [
      `chapter: ${requested}`,
      `mode:    答案`,
      `output:  ${output}`,
      "",
      `${output}/LEARNING.md`,
      "",
    ].join("\n"),
  );
} catch (error) {
  if (createdOutput) {
    await rm(output, { recursive: true, force: true });
  }
  throw error;
} finally {
  // 无论成功失败都清理 worktree
  execFileSync("git", ["worktree", "remove", "--force", worktree], {
    cwd: repoRoot,
    stdio: "pipe",
  });
  await rm(worktree, { recursive: true, force: true });
}
