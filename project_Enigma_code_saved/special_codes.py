import sys
import os
custom_folder=r'./code_safe'
# 获取 custom_folder 的绝对路径
custom_folder_path = os.path.abspath(custom_folder)

# 将 custom_folder 添加到 sys.path
if custom_folder_path not in sys.path:
    sys.path.append(custom_folder_path)

# 导入 custom_folder 下的模块

from enigma_encoder.machine import EnigmaMachine

Machine = EnigmaMachine(wheels_dir=op('current_wheel_path').text)

df=Machine.show_encoded_letter_without_turn('a')

op('text6').text=df
