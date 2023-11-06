import sqlite3
from flask import Flask, render_template, Response, request, abort, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500/collect", "http://127.0.0.1:5000"])
class SQLQueries:
    def __init__(self):
        self.conn = sqlite3.connect("static/database/info.db")
        self.curr = self.conn.cursor()
    def search_name(self, id):
        self.curr.execute('SELECT * FROM site_data WHERE id=?', (id,))
        results = self.curr.fetchall()
        return (results[0][1], results[0][2], results[0][3])
@app.errorhandler(404)
def page_not_found(e):
    return render_template(r"error404.html")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    data = request.json
    print(data)
    return "Data printed!", 200

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    id = ""
    if request.method == 'POST':
        id = request.form.get('id', '')
        if int(id)<=551:
            name, site_code, pages = SQLQueries().search_name(id)
            return render_template('analysis.html', name=site_code, pages=pages)
        else:
            abort(404)
    else:
        abort(404)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template(r"homepage.html")

if __name__ == '__main__':
    app.run(debug=True)