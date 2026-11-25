import ast
import runpy
import subprocess
import sys
from itertools import product
from pathlib import Path
from types import ModuleType

"""
REG / ROB 網路架構批次訓練。

將 LAYERS 和 WIDTHS 交叉組合，共測試 9 種架構。
每組先完成 joint_reg.py,再用同一組參數執行 joint_rob.py,
讓 ROB 從對應的 REG 權重初始化。

每次訓練都在獨立的 Python 程序執行。子程序先讀取 settings.py,
在記憶體中代入這一輪的 LAYER 和 NET,再執行設定與訓練。
因此網路架構、權重路徑和收斂資料路徑都會使用同一組 NN_TAG,
磁碟上的 settings.py 仍保留原本的設定。

任一訓練程序報錯時，排程會停止。
重新啟動會從清單第一組開始；重跑同一組會寫入相同目錄。
"""

PROJECT_DIR = Path(__file__).resolve().parent
LAYERS = (3, 5, 7)
WIDTHS = (256, 512, 1024)


def run_training(script_name, layer, net):
    settings_path = PROJECT_DIR / "settings.py"
    source = ast.parse(settings_path.read_bytes(), filename=str(settings_path))

    # 只替換這次程序在記憶體中的兩個設定值。
    overrides = {"LAYER": layer, "NET": net}
    replaced = set()

    for statement in source.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue

        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in overrides:
            statement.value = ast.copy_location(
                ast.Constant(value=overrides[target.id]), statement.value
            )
            replaced.add(target.id)

    if replaced != set(overrides):
        missing = ", ".join(sorted(set(overrides) - replaced))
        raise RuntimeError(f"settings.py 找不到設定：{missing}")

    settings_module = ModuleType("settings")
    settings_module.__file__ = str(settings_path)
    sys.modules["settings"] = settings_module
    exec(compile(source, str(settings_path), "exec"), vars(settings_module))

    script_path = PROJECT_DIR / script_name
    sys.argv = [str(script_path)]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        script_name, layer, net = sys.argv[2:]
        run_training(script_name, int(layer), int(net))
    else:
        cases = list(product(LAYERS, WIDTHS))

        for index, (layer, net) in enumerate(cases, start=1):
            tag = f"L{layer:02d}_NET{net}"

            for script_name in ("joint_reg.py", "joint_rob.py"):
                print(f"\n[{index}/{len(cases)}] {tag} | {script_name}", flush=True)

                subprocess.run(
                    [
                        sys.executable, "-u", str(Path(__file__).resolve()),
                        "--worker", script_name, str(layer), str(net),
                    ],
                    cwd=PROJECT_DIR,
                    check=True,
                )

            print(f"[{index}/{len(cases)}] {tag} 完成", flush=True)