from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from ...core import config
from ...services.vision_pipeline import process_image

router = APIRouter()


@router.post("/analyze")
async def analyze_skin_test(file: UploadFile = File(...)):
	# Validate content type
	if file.content_type not in config.ALLOWED_CONTENT_TYPES:
		raise HTTPException(status_code=400, detail="Invalid file type")

	contents = await file.read()
	if len(contents) > config.MAX_UPLOAD_SIZE:
		raise HTTPException(status_code=413, detail="File too large")

	try:
		results = process_image(contents)
		return JSONResponse(content=results)
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc))
	except Exception as exc:
		raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze-viewer")
async def analyze_with_viewer(file: UploadFile = File(...)):
	"""Analyze image and return an HTML page displaying the annotated and segmented images."""
	# Validate content type
	if file.content_type not in config.ALLOWED_CONTENT_TYPES:
		raise HTTPException(status_code=400, detail="Invalid file type")

	contents = await file.read()
	if len(contents) > config.MAX_UPLOAD_SIZE:
		raise HTTPException(status_code=413, detail="File too large")

	try:
		results = process_image(contents)
		
		annotated = results["visualization"]["annotated"]
		segmented = results["visualization"]["segmented"]
		
		# Extract wheal data for display
		wheals_html = ""
		for w in results["results"]:
			wheals_html += f"""
			<tr>
				<td>{w['id']}</td>
				<td>{w['diameter_mm']} mm</td>
				<td>{w['severity']}</td>
				<td>{w['confidence']:.2f}</td>
			</tr>
			"""
		
		html_content = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<title>Allergy Wheal Analysis</title>
			<style>
				body {{
					font-family: Arial, sans-serif;
					margin: 20px;
					background-color: #f5f5f5;
				}}
				.container {{
					max-width: 1400px;
					margin: 0 auto;
					background-color: white;
					padding: 20px;
					border-radius: 8px;
					box-shadow: 0 2px 4px rgba(0,0,0,0.1);
				}}
				h1 {{
					color: #333;
					text-align: center;
				}}
				.images-grid {{
					display: grid;
					grid-template-columns: 1fr 1fr;
					gap: 20px;
					margin: 20px 0;
				}}
				.image-container {{
					border: 1px solid #ddd;
					border-radius: 4px;
					padding: 10px;
					background-color: #fafafa;
				}}
				.image-container h3 {{
					margin-top: 0;
					color: #555;
				}}
				img {{
					max-width: 100%;
					height: auto;
					border-radius: 4px;
				}}
				table {{
					width: 100%;
					border-collapse: collapse;
					margin-top: 20px;
				}}
				th, td {{
					padding: 10px;
					text-align: left;
					border-bottom: 1px solid #ddd;
				}}
				th {{
					background-color: #4CAF50;
					color: white;
				}}
				tr:hover {{
					background-color: #f5f5f5;
				}}
				.meta {{
					margin: 20px 0;
					padding: 10px;
					background-color: #e8f4f8;
					border-left: 4px solid #2196F3;
					border-radius: 4px;
				}}
			</style>
		</head>
		<body>
			<div class="container">
				<h1>Allergy Wheal Analysis Results</h1>
				
				<div class="meta">
					<p><strong>Processed at:</strong> {results['meta']['processed_at']}</p>
					<p><strong>Scale (PPM):</strong> {results['calibration']['scale_ppm']:.2f} pixels/mm</p>
					<p><strong>Total Wheals Detected:</strong> {len(results['results'])}</p>
				</div>
				
				<div class="images-grid">
					<div class="image-container">
						<h3>Annotated Image (Green Circles = Detected Wheals)</h3>
						<img src="{annotated}" alt="Annotated Image">
					</div>
					<div class="image-container">
						<h3>Segmentation Mask (Red = Detected Regions)</h3>
						<img src="{segmented}" alt="Segmented Image">
					</div>
				</div>
				
				<h2>Wheal Measurements</h2>
				<table>
					<thead>
						<tr>
							<th>ID</th>
							<th>Diameter (mm)</th>
							<th>Severity</th>
							<th>Confidence</th>
						</tr>
					</thead>
					<tbody>
						{wheals_html if wheals_html else "<tr><td colspan='4' style='text-align: center;'>No wheals detected</td></tr>"}
					</tbody>
				</table>
			</div>
		</body>
		</html>
		"""
		return HTMLResponse(content=html_content)
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc))
	except Exception as exc:
		raise HTTPException(status_code=500, detail="Internal server error")