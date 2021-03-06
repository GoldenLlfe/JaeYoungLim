# -*- coding: utf-8 -*-
"""Project_1_Final

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qAT9jVhAvg2XYyUhm4XiQmPnbpZ-uo2h
"""

!pip install mediapipe
import numpy as np
import cv2
import mediapipe as mp
import joblib
from collections import Counter
import math
#from files.unicode import join_jamos
from PIL import ImageFont, ImageDraw, Image
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab.patches import cv2_imshow

__all__ = ["split_syllable_char", "split_syllables",
           "join_jamos", "join_jamos_char",
           "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.
    Arguments:
        c: A Korean character.
    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.
    Example:
        >>> split_syllable_char("???")
        ("???", "???", "???")
        >>> split_syllable_char("???")
        ("???", "???", None)
        >>> split_syllable_char("???")
        (None, "???", None)
        >>> split_syllable_char("???")
        ("???", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.
    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)
    Returns:
        Hangul-split string
    Example:
        >>> split_syllables("???????????????")
        "????????????????????????????????????"
        >>> split_syllables("???????????????~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("??????????????????", pad="x")
        '????????????????????????x??????x??????xx???x'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))


def join_jamos_char(init, med, final=None):
    """
    Combines jamos into a single syllable.
    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)
    Returns:
        A Korean syllable.
    """
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.
    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)
    Returns:
        A string
    Example:
        >>> join_jamos("????????????????????????????????????")
        "???????????????"
        >>> join_jamos("???????????????????????????????????????")
        "??????????????????"
        >>> join_jamos()
    """
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None     
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string

def korean_plot(image, text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font=ImageFont.truetype('/content/drive/MyDrive/Project_1/NanumMyeongjo-YetHangul.ttf', 100)
    # font = ImageFont.load_default()
    org = (200,150)
    draw.text(org,text,font = font, fill = (150,0,255))
    img = np.array(img)
    return img

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

"""# ??? ??????"""

from IPython.display import Image as Img
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Img(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

# In[10]:
kn = joblib.load('/content/drive/MyDrive/Project_1/ML-model.pkl')
print("start...")
image = cv2.imread('./{}'.format(filename))
w = 640
h = 480
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
feature_list = []

my_char = ['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
           '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', 'a', 'b', 'c']
dot_list = [4, 8, 12, 14, 16, 18, 20]
ja = ['???','???','???','???','???','???','???','???','???','???','???','???','???','???']
mo = ['???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???']
mo1 = ['???', '???']
mo2 = ['???', '???']
mo3 = ['???', '???']
mo4 = ['???', '???', '???', '???']
ssang = ['???','???','???','???','???']
jong = [['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'],
        ['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'], ['???', '???', '???'],
        ['???', '???', '???']]
checker1 = 0
checker2 = 0
temp_ch = ''
my_word = ''
sentence = ''
dump_list = []
image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
image.flags.writeable = False
results = hands.process(image)
image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        mean_x = hand_landmarks.landmark[0].x  # x??? ?????? 0??? ??? 1??? ???
        mean_y = hand_landmarks.landmark[0].y  # y??? ????????? 0??? ?????? 1??? ?????????
        min_x = w - 1; max_x = 0.0; min_y = h - 1; max_y = 0.0
        for i in range(0, 21):  # ????????????
            hlm = hand_landmarks.landmark[i]
            if hlm.x * w > max_x:
                max_x = hlm.x * w
            if hlm.x * w < min_x:
                min_x = hlm.x * w
            if hlm.y * h > max_y:
                max_y = hlm.y * h
            if hlm.y * h < min_y:
                min_y = hlm.y * h
        for i in dot_list:
            hlm = hand_landmarks.landmark[i]
            feature_list.append(((hlm.x - mean_x) * w) / (max_x - min_x))
            feature_list.append((hlm.y - mean_y) * h / (max_y - min_y))
        d8 = hand_landmarks.landmark[8]
        d12 = hand_landmarks.landmark[12]
        d16 = hand_landmarks.landmark[16]
        d23 = math.sqrt(pow(d8.x * w - d12.x * w, 2) + pow(d8.y * h - d12.y * h, 2))
        d34 = math.sqrt(pow(d16.x * w - d12.x * w, 2) + pow(d16.y * h - d12.y * h, 2))
        feature_list.append(d23 / d34 - 1)
        #feature_list.append((max_y - min_y) / (max_x - min_x) - 1)
        feature_list = np.round(feature_list, decimals=5)
        feature_list = [feature_list]
        proba = kn.predict_proba(feature_list)
        C = my_char[np.argmax(proba[0])]
        dump_list.append(C)
        feature_list = []
if True:
  ch = Counter(dump_list).most_common()[0][0]
  if ch != temp_ch:
      if ch in ja:  # ????????????
          for i in range(0, 11):
              if ch == jong[i][1] and temp_ch == jong[i][0]:
                  my_word = my_word[:-1]
                  my_word += jong[i][2]
                  checker1 = 1
                  checker2 = 1
                  break
          if checker1 == 0:  # ??????????????? ????????? ?????????
              checker2 = 0
              my_word += ch
          temp_ch = ch
          sentence = join_jamos(my_word)
          print(sentence)
          checker1 = 0
      elif ch in mo:  # ????????????
          if temp_ch == '???' and ch in mo2:  # ????????? ???or?????????
              my_word = my_word[:-1]
              temp_ch = ch
              ch = mo4[mo2.index(ch)]
          elif temp_ch == '???' and ch in mo3:  # ????????? ???or?????????
              my_word = my_word[:-1]
              temp_ch = ch
              ch = mo4[mo3.index(ch) + 2]
          else:  # ?????? ??????
              if checker2 == 1:
                  l = my_word[-1]
                  my_word = my_word[:-1]
                  for i in range(0, 11):
                      if jong[i][2] == l:
                          my_word += jong[i][0]
                          my_word += jong[i][1]
                          break
                  temp_ch = ch
                  checker2 = 0
              else:
                  temp_ch = ch
          my_word += ch
          sentence = join_jamos(my_word)
          print(sentence)

      else:  # a, b, c or d??????
          checker2 = 0
          if ch == 'a':
              if temp_ch in ssang:
                  my_word = my_word[:-1]
                  my_word += chr(ord(temp_ch) + 1)
          elif ch == 'c':
              my_word = my_word[:-1]
          elif ch == 'd':
              my_word += ' '
          sentence = join_jamos(my_word)
          print(sentence)
          temp_ch = ch
  dump_list = []


image = korean_plot(image, sentence)
# cv2.putText(image, sentence, (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)

cv2_imshow(image)

hands.close()
cv2.waitKey()
cv2.destroyAllWindows()



