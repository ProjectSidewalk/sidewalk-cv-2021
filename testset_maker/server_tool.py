import http.server
import io
import os
import csv
import urllib
import posixpath
from http import HTTPStatus

# Instructions for use:
# 1. If ssh-ing, use 'ssh -L local_port:remote_ip:remote_port user@hostname.com'.
#    For example, 'ssh -L 40296:rainbowdash:40296 shokiami@attu.cs.washington.edu'.
#    Make sure the port address matches `PORT`.
# 2. If test_set.csv doesn't exist yet, run init_testset.py.
# 3. Run server_tool.py.
# 4. Navigate to 'http://localhost:40296/testset_maker' in your browser and begin editing!
#    Click through crops with "Next" and "Prev". 
#    Clicking "Save" saves and goes to the next crop.
# 5. Use Ctrl+C to exit.

PORT = 40296
TESTSET_PATH = '../datasets/test_set.csv'
ROOT_DIRECTORY = '/tmp/datasets'
CROPS_DIRECTORY = '/crops/'

csv_in = open(TESTSET_PATH, 'r')
csv_reader = csv.reader(csv_in)
next(csv_reader)
csv_list = []
label_ids_to_csv_indices = dict()
for index, row in enumerate(csv_reader):
  filename = row[0]
  label_ids = eval(row[1])
  csv_list.append([filename, label_ids])

  # different filename formats that are accepted by search tool
  acceptable_filenames = [filename]

  # without extension
  acceptable_filenames.append(filename[:filename.index('.jpg')]) 
  # without _0 size indicator at end or extension
  acceptable_filenames.append(filename[:filename.index("_0")])
  # without city prefix or extension
  acceptable_filenames.append(filename[filename.index("/") + 1:filename.index(".jpg")])
  # without city prefix or _0 size indicator at end or extension
  acceptable_filenames.append(filename[filename.index("/") + 1:filename.index("_0")])

  for name in acceptable_filenames:
    label_ids_to_csv_indices[name] = index

def save_to_file():
  csv_out = open(TESTSET_PATH, 'w')
  csv_writer = csv.writer(csv_out)
  csv_writer.writerow(['image_name', 'label_type'])
  for row in csv_list:
    csv_writer.writerow(row)

class MyHTTPHandler(http.server.SimpleHTTPRequestHandler):

  def __init__(self, *args):
    http.server.SimpleHTTPRequestHandler.__init__(self, *args)

  def translate_path(self, path):
    path = path.split('?',1)[0]
    path = path.split('#',1)[0]
    trailing_slash = path.rstrip().endswith('/')
    try:
        path = urllib.parse.unquote(path, errors='surrogatepass')
    except UnicodeDecodeError:
        path = urllib.parse.unquote(path)
    path = posixpath.normpath(path)
    words = path.split('/')
    words = filter(None, words)
    path = ROOT_DIRECTORY
    for word in words:
        if os.path.dirname(word) or word in (os.curdir, os.pardir):
            continue
        path = os.path.join(path, word)
    if trailing_slash:
        path += '/'
    return path

  def send_head(self):
    if '/save' in self.path:
      index = int(self.path[self.path.index("?") - 1])
      form_data = self.path.partition('?')[2]
      if form_data != '':
        label_ids = list(map(int, form_data.replace('=on','').split('&')))
      else:
        label_ids = []
      csv_list[index][1] = label_ids
      save_to_file()
      self.send_response(HTTPStatus.MOVED_PERMANENTLY)
      self.send_header('Location', '/testset_maker/' + str(index + 1))
      self.send_header('Cache-Control', 'no-store')
      self.send_header('Content-Length', '0')
      self.end_headers()
      return None
    elif (self.path == '/testset_maker' or self.path == '/testset_maker/'):
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header('Location', '/testset_maker/0')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None
    elif '/testset_maker' in self.path:
      # one-indexed
      if "search=" in self.path:
        searched_label_id = urllib.parse.unquote(self.path[self.path.index("=") + 1:])
        if searched_label_id in label_ids_to_csv_indices.keys():
          index = label_ids_to_csv_indices[searched_label_id]
        else:
          index = 0
      else:
        index = int(self.path.replace('/testset_maker/', '').partition('?')[0])
      img_id = csv_list[index][0].replace('.jpg', '')

      text = """
      <h2 style="display: flex; justify-content: center; margin-top: 25px;">Crop #{}/{}</h2>
      <h3 style="display: flex; justify-content: center;">Label ID: {}</h3>
      <div style="display: flex; justify-content: center;">
        <img src="{}{}.jpg" width="500" height="500"></img>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <a href="/testset_maker/{}" style="margin-right: 125px;">Prev</a>
        <form method="get" action="/save{}">
          <div>
            <input type="checkbox" id="1" name="1" {}>
            <label for="1"> Curb Ramp</label><br>
          </div>
          <div>
            <input type="checkbox" id="2" name="2" {}>
            <label for="2"> Missing Curb Ramp</label><br>
          </div>
          <div>
            <input type="checkbox" id="3" name="3" {}>
            <label for="3"> Obstacle</label>
          </div>
          <div>
            <input type="checkbox" id="4" name="4" {}>
            <label for="4"> Surface Problem</label>
          </div>
          <div style="margin-top: 5px;">
            <input type="submit" value="Save">
          </div>
        </form>
        <a href="/testset_maker/{}" style="margin-left: 125px;">Next</a>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <form action="/testset_maker">
            Search by label ID:
            <input type="text" name="search" id="search">
            <button type="submit" id="go">go</button>
        </form>
      </div>
      """.format(
        index + 1,
        len(csv_list),
        img_id,
        CROPS_DIRECTORY,
        img_id, 
        index - 1 if index > 0 else index,
        index,
        'checked' if csv_list[index][1].count(1) else '', 
        'checked' if csv_list[index][1].count(2) else '', 
        'checked' if csv_list[index][1].count(3) else '', 
        'checked' if csv_list[index][1].count(4) else '', 
        index + 1 if index < len(csv_list) - 1 else index
      )
      encoded = text.encode('utf-8')
      f = io.BytesIO()
      f.write(encoded)
      f.seek(0)
      self.send_response(HTTPStatus.OK)
      self.send_header('Content-type', 'text/html; charset=utf-8')
      self.send_header('Cache-Control', 'no-store')
      self.send_header('Content-Length', str(len(encoded)))
      self.end_headers()
      return f
    else:
      return http.server.SimpleHTTPRequestHandler.send_head(self)

Handler = MyHTTPHandler

with http.server.HTTPServer(('', PORT), Handler) as httpd:
  print('Serving at port', PORT)
  httpd.serve_forever()
