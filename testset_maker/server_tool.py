import http.server
import io
import os
import csv
import urllib
import posixpath
from http import HTTPStatus

# Instructions for use:
# 1. If ssh-ing, use 'ssh -L local_port:remote_ip:remote_port user@hostname.com'.
#    For example, 'ssh -L 40296:rainbowdash:40296 shokiami@attu.cs.washington.edu'
#    Make sure the port address matches `PORT`.
# 2. If test_set.csv doesn't exist yet, run init_testset.py.
# 3. Run server_tool.py.
# 4. Go to 'http://localhost:40296/testset_maker' in your browser and begin editing!
#    Navigate crops with the "Next" and "Prev" buttons. 
#    Clicking "Save" saves and goes to the next crop.
# 5. Use Ctrl+C to exit.

PORT = 40296
TESTSET_PATH = '../datasets/test_set.csv'
DIRECTORY = '/tmp/datasets/'

csv_in = open(TESTSET_PATH, 'r')
csv_reader = csv.reader(csv_in)
next(csv_reader)
csv_list = []
for row in csv_reader:
  filename = row[0]
  label_ids = eval(row[1])
  csv_list.append([filename, label_ids])

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
    # abandon query parameters
    path = path.split('?',1)[0]
    path = path.split('#',1)[0]
    # Don't forget explicit trailing slash when normalizing. Issue17324
    trailing_slash = path.rstrip().endswith('/')
    try:
        path = urllib.parse.unquote(path, errors='surrogatepass')
    except UnicodeDecodeError:
        path = urllib.parse.unquote(path)
    path = posixpath.normpath(path)
    words = path.split('/')
    words = filter(None, words)
    path = DIRECTORY
    for word in words:
        if os.path.dirname(word) or word in (os.curdir, os.pardir):
            # Ignore components that are not a simple file/directory name
            continue
        path = os.path.join(path, word)
    if trailing_slash:
        path += '/'
    return path

  def send_head(self):
    if (self.path == '/testset_maker'):
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header('Location', '/testset_maker/1')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None
    if (self.path.count('/testset_maker/')):
      index = int(self.path.replace('/testset_maker/', '').partition('?')[0])
      img_id = csv_list[index - 1][0].replace('.jpg', '')
      if self.path.count('?'):
        form_data = self.path.partition('?')[2]
        if form_data != '':
          label_ids = list(map(int, form_data.replace('=on','').split('&')))
        else:
          label_ids = []
        csv_list[index - 1][1] = label_ids
        save_to_file()
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header('Location', '/testset_maker/' + str(index + 1))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return None
      text = """
      <h2 style="display: flex; justify-content: center; margin-top: 25px;">Crop #{}/{}</h2>
      <h3 style="display: flex; justify-content: center;">Label ID: {}</h3>
      <div style="display: flex; justify-content: center;">
        <img src="/crops/{}_0.jpg" width="500" height="500"></img>
      </div>
      <div style="display: flex; justify-content: center; margin-top: 15px;">
        <a href="/testset_maker/{}" style="margin-right: 125px;">Prev</a>
        <form method="get">
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
      """.format(
        index,
        len(csv_list),
        img_id,
        img_id, 
        index-1 if index > 1 else index, 
        'checked' if csv_list[index - 1][1].count(1) else '', 
        'checked' if csv_list[index - 1][1].count(2) else '', 
        'checked' if csv_list[index - 1][1].count(3) else '', 
        'checked' if csv_list[index - 1][1].count(4) else '', 
        index+1 if index < len(csv_list) else index
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
