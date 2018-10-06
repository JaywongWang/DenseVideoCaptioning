import os
import sys
import subprocess
import threading


meteor_cmd = ['java', '-jar', '-Xmx2G', 'meteor-1.5.jar', '-', '-', '-stdio', '-l', 'en', '-norm']
# meteor_cmd = ['echo']
score_line = b'SCORE ||| an older building with birds flying near it ||| a church with a high spire and decorative roof tiles ||| a large clock tower towering over a small city ||| picture of a church and its tall steeple ||| birds fly around a tall building with a clock ||| a a kitchen\n'

# meteor_cmd = ['cat']
# score_line = b'test'
meteor_p = subprocess.Popen(meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE,bufsize=0)


meteor_p.stdin.write(score_line)
print('waitiing')
result = meteor_p.stdout.readline().decode().strip()
print(result)
