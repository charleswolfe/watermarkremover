
import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import uuid

from src.handlers.process_video import handle_process_video


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

jobs = {}



# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')




# --- ENDPOINT 1: Video Upload ---
# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'video_file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#
#     file = request.files['video_file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
#
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#
#     # Return the path so the frontend knows which file to process
#     return jsonify({
#         "status": "uploaded",
#         "filename": filename,
#         "server_path": filepath
#     })


@app.route('/status/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route('/process', methods=['POST'])
def process_video():
    job_id = str(uuid.uuid4())
    video_file = request.files['video']
    upscale = True if  request.form['upscale'] == "true" else False
    use_cpu = True if  request.form['use_cpu'] == "true" else False

    rectangles = json.loads(request.form['rectangles'])

    # Save video
    video_path = f"uploads/{video_file.filename}"
    video_file.save(video_path)
    base, ext = os.path.splitext(video_file.filename)
    # Construct the new filename: base + '-clean' + extension (e.g., 'video-clean.mp4')
    output_video = f"{base}-clean{ext}"
    output_path = f"/app/output/{output_video}"
    download_path = f"/download/{output_video}"

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_url": None,
        "error": None
    }

    #process wm
    # handle_process_video(
    #     input_path=video_path,
    #     output_path=output_path,
    #     use_cpu=use_cpu,
    #     upscale=upscale,
    #     rectangles=rectangles
    # )

    def run_job():
        try:
            jobs[job_id]["status"] = "processing"
            handle_process_video(
                input_path=video_path,
                output_path=output_path,
                use_cpu=use_cpu,
                upscale=upscale,
                rectangles=rectangles,
                progress_cb=lambda p: jobs[job_id].update(progress=p)
            )
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["output_url"] = download_path
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)

    threading.Thread(target=run_job, daemon=True).start()


# todo delete uploaded files



    # return jsonify({
    #     'success': True,
    #     'message': 'Video processed',
    #     'output_url': output_path
    # })

    return jsonify({
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
        })
@app.route('/download/<filename>')
def download_file(filename):
    """Serve processed video for download"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='video/mp4'
    )

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
