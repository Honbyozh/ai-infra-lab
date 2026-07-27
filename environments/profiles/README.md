# 本机角色配置

每台机器只需初始化一次本机角色。根目录的 `.machine.local.yaml` 是本机事实源，已被 `.gitignore` 忽略，不会在两台机器之间互相覆盖。

学习机执行：

```bash
cp environments/profiles/learning-machine.yaml.example .machine.local.yaml
```

实验机执行：

```bash
cp environments/profiles/experiment-machine.yaml.example .machine.local.yaml
```

复制后按实际情况修改 `machine_id` 和 `description`。角色规则由根目录 `AGENTS.md` 定义；以后进入仓库工作的 Agent 应先读取本机配置。

注意：机器角色只描述能力边界，不覆盖用户授权、安全规则和任务验收标准。学习机不能用理论结论代替真实实验；需要实验的候选方向体验留给实验机或后续云环境完成。
