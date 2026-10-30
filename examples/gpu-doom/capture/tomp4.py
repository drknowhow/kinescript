"""tomp4.py RUN.out OUT.mp4: palettized 320x200 frames from play.html -> 35 fps H.264, 1600x1200 nearest."""
import json, sys, base64, subprocess, numpy as np
s = open(sys.argv[1]).read(); r = json.loads(s[s.find('{'):])
if 'error' in r: sys.exit(r['error'])
pal = np.array([[c & 255, (c >> 8) & 255, (c >> 16) & 255] for c in r['palette']], np.uint8)
ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '320x200', '-r', '35', '-i', '-',
                       '-vf', 'scale=1600:1200:flags=neighbor', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
                       '-pix_fmt', 'yuv420p', sys.argv[2]], stdin=subprocess.PIPE)
for f in r['frames']:
    ff.stdin.write(pal[np.frombuffer(base64.b64decode(f), np.uint8).reshape(200, 320)].tobytes())
ff.stdin.close(); ff.wait()
print(sys.argv[2], len(r['frames']), 'frames', {k: r.get(k) for k in ['map', 'result', 'tics', 'kills', 'totalKills', 'items', 'health']})
print('events', [(e['tic'], e['ev'], e.get('kind')) for e in r['events']][:40])
