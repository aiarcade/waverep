import sys
import os
import random
from pydub import AudioSegment
sound = AudioSegment.from_mp3(sys.argv[1])
origin = 0
print(sound)
# Generate three repetitions
r = random.sample(range(2, 10), 3)
r.sort()
r1 = r[0]
r2 = r[1]
r3 = r[2]

s = random.sample(range(3,7), 3)
s.sort()
s1 = s[0]
s2 = s[1]
s3 = s[2]

split_point1 = len(sound) / r3
split_point2 = len(sound) / r2
split_point3 = len(sound) / r1

print("S and R is",s,r)
first_half = sound[:split_point1]
second_half = sound[split_point1 - (split_point1/s1):split_point2]
third_half = sound[split_point2 - (split_point2/s2):split_point3]
fourth_half = sound[split_point3 - (split_point3/s3):]
print("Breakpoints are ")
print(origin, split_point1)
print(split_point1 - (split_point1/s1), split_point2)
print(split_point2 - (split_point2/s2), split_point3)
print(split_point3 - (split_point3/s3), len(sound))
# The break points can be found from the array indices of first_half and second_half
# There can actually be more than two halves too, like 4 or 5 halves. For n halves we will have (n-1) break points

#generating transformed audio
transformedAudio = first_half + second_half + third_half + fourth_half 
transformedAudio.export(sys.argv[2], format="wav")
sound.export(sys.argv[3],format="wav")
