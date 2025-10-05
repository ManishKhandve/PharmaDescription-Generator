import os
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from io import BytesIO
import logging

# Import custom modules
from llm_client import LLMClient, BatchProcessor
from utils import ExcelHandler, ProgressTracker, FileValidator, estimate_processing_time

# Configure logging with comprehensive error handling
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
except Exception as e:
    print(f"Error setting up logging: {str(e)}")
    logger = logging.getLogger(__name__)

# Initialize Flask app with error handling
try:
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'pharma-desc-gen-2024-secure-key'
    
    # Use absolute paths to ensure directories are created in the correct location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'output')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Ensure directories exist with error handling
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
        logger.info(f"Directories created: {app.config['UPLOAD_FOLDER']}, {app.config['OUTPUT_FOLDER']}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        # Create fallback directories
        app.config['UPLOAD_FOLDER'] = 'uploads'
        app.config['OUTPUT_FOLDER'] = 'output'
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
except Exception as e:
    logger.error(f"Critical error initializing Flask app: {str(e)}")
    raise


# Global dictionary to track processing jobs with error handling
try:
    processing_jobs = {}
    logger.info("Processing jobs dictionary initialized")
except Exception as e:
    logger.error(f"Error initializing processing jobs: {str(e)}")
    processing_jobs = {}

# Inactivity timer logic
last_activity = time.time()
INACTIVITY_TIMEOUT = 5 * 60  # 5 minutes

def is_processing():
    # Return True if any job is currently processing
    for job in processing_jobs.values():
        if job.status in ("processing", "initializing"):
            return True
    return False

def update_activity():
    global last_activity
    last_activity = time.time()

def inactivity_watcher():
    while True:
        time.sleep(10)
        idle_time = time.time() - last_activity
        logger.debug(f"[InactivityWatcher] is_processing={is_processing()} | idle_time={idle_time:.1f}s | timeout={INACTIVITY_TIMEOUT}s")
        if not is_processing() and (idle_time > INACTIVITY_TIMEOUT):
            logger.info("No activity for 5 minutes. Shutting down server...")
            os._exit(0)

threading.Thread(target=inactivity_watcher, daemon=True).start()


class ProcessingJob:
    """
    Represents a single processing job with progress tracking and original data preservation.
    """
    
    def __init__(self, job_id: str, total_products: int, original_data=None):
        self.job_id = job_id
        self.total_products = total_products
        self.processed_count = 0
        self.current_product = ""
        self.status = "initializing"  # initializing, processing, completed, failed, stopped
        self.progress_percentage = 0
        self.start_time = datetime.now()
        self.end_time = None
        self.output_file = None
        self.error_message = None
        self.results = []
        self.stop_requested = False
        self.original_data = original_data  # Store original DataFrame
    
    def update_progress(self, processed: int, current: str):
        """Update job progress."""
        self.processed_count = processed
        self.current_product = current
        self.progress_percentage = min(100, int((processed / self.total_products) * 100))
        
    def complete(self, output_file: str):
        """Mark job as completed."""
        self.status = "completed"
        self.progress_percentage = 100
        self.end_time = datetime.now()
        self.output_file = output_file
        
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error
    
    def stop(self, output_file: str = None):
        """Mark job as stopped by user."""
        self.status = "stopped"
        self.end_time = datetime.now()
        self.stop_requested = True
        if output_file:
            self.output_file = output_file


@app.before_request
def before_request():
    update_activity()

@app.route('/')
def index():
    """
    Main page with upload form and processing interface.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and initiate processing.
    """
    logger.info("=== UPLOAD REQUEST RECEIVED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request form keys: {list(request.form.keys())}")
    logger.info(f"Request files keys: {list(request.files.keys())}")
    logger.info(f"Model type: {request.form.get('model_type', 'NOT PROVIDED')}")
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Get form data
        api_key = request.form.get('api_key', '').strip()
        model_type = request.form.get('model_type', 'mistral').lower()
        
        # API key is required for all cloud models
        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'})
        
        openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
        allowed_models = openrouter_models + ['gemini']
        if model_type not in allowed_models:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'Invalid model type',
                'details': 'Supported models: Mistral, OpenChat, DeepSeek, GPT-OSS (OpenRouter) and Gemini.'
            })
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.xlsx', '.xls')):
            return jsonify({'success': False, 'error': 'Please upload an Excel file (.xlsx or .xls)'})
        
        job_id = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(upload_path)
        
        # Validate file
        validation_result = FileValidator.validate_input_file(upload_path)
        if not validation_result['valid']:
            os.remove(upload_path)  # Clean up
            return jsonify({'success': False, 'error': validation_result['error']})
        
        # Create processing job
        job = ProcessingJob(job_id, validation_result['product_count'])
        processing_jobs[job_id] = job
        
        # Get time estimate
        time_estimate = estimate_processing_time(validation_result['product_count'], model_type)
        
        # Start processing in background thread
        threading.Thread(
            target=process_file_async,
            args=(job_id, upload_path, api_key, model_type),
            daemon=True
        ).start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'product_count': validation_result['product_count'],
            'estimated_time': time_estimate['formatted_time'],
            'recommendation': time_estimate['recommendation']
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})



@app.route('/progress/<job_id>')
def get_progress(job_id):
    """
    Get progress information for a processing job.
    """
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    elapsed_time = ""
    estimated_time = "Calculating..."
    if job.start_time:
        elapsed = datetime.now() - job.start_time
        elapsed_seconds = int(elapsed.total_seconds())
        # Always show elapsed time as H:M:S
        elapsed_h = elapsed_seconds // 3600
        elapsed_m = (elapsed_seconds % 3600) // 60
        elapsed_s = elapsed_seconds % 60
        elapsed_time = f"{elapsed_h}h {elapsed_m}m {elapsed_s}s"

        # Estimate time left if possible
        if job.processed_count and job.total_products and job.processed_count > 0:
            avg_time_per = elapsed_seconds / job.processed_count
            remaining = job.total_products - job.processed_count
            est_seconds = int(avg_time_per * remaining)
            est_h = est_seconds // 3600
            est_m = (est_seconds % 3600) // 60
            est_s = est_seconds % 60
            estimated_time = f"{est_h}h {est_m}m {est_s}s"
        else:
            estimated_time = "Estimating..."

    # Try to get failed_count and rate_limit_count if available
    failed_count = 0
    rate_limit_count = 0
    if hasattr(job, 'failed_count'):
        failed_count = job.failed_count
    elif hasattr(job, 'results') and isinstance(job.results, list):
        failed_count = len([r for r in job.results if isinstance(r, dict) and r.get('status') == 'failed'])

    if hasattr(job, 'rate_limit_count'):
        rate_limit_count = job.rate_limit_count

    return jsonify({
        'status': job.status,
        'progress': job.progress_percentage,
        'processed_count': job.processed_count,
        'total_count': job.total_products,
        'current_product': job.current_product,
        'elapsed_time': elapsed_time,
        'estimated_time': estimated_time,
        'error_message': job.error_message,
        'can_download': job.status in ['completed', 'stopped'] and job.output_file is not None,
        'failed_count': failed_count,
        'rate_limit_count': rate_limit_count
    })


@app.route('/stop/<job_id>', methods=['POST'])
def stop_processing(job_id):
    """
    Stop processing job and prepare partial results for download.
    """
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    if job.status not in ['processing', 'initializing']:
        return jsonify({'error': 'Job cannot be stopped'}), 400
    
    # Mark job for stopping
    job.stop_requested = True
    
    logger.info(f"Stop requested for job {job_id}")
    
    return jsonify({
        'success': True,
        'message': 'Stop request sent. Processing will halt after current batch.',
        'processed_count': job.processed_count
    })


@app.route('/download/<job_id>')
def download_result(job_id):
    """
    Download the generated output file.
    """
    job = processing_jobs.get(job_id)
    if not job or job.status not in ['completed', 'stopped'] or not job.output_file:
        return jsonify({'error': 'File not ready for download'}), 404
    
    if not os.path.exists(job.output_file):
        return jsonify({'error': 'Output file not found'}), 404
    
    # Generate download filename
    status_suffix = "_partial" if job.status == "stopped" else ""
    download_name = f"product_descriptions_{job_id[:8]}{status_suffix}.xlsx"
    
    return send_file(
        job.output_file,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/status')
def system_status():
    """
    Get system status and active jobs count.
    """
    active_jobs = len([j for j in processing_jobs.values() if j.status == 'processing'])
    completed_jobs = len([j for j in processing_jobs.values() if j.status == 'completed'])
    failed_jobs = len([j for j in processing_jobs.values() if j.status == 'failed'])
    
    return jsonify({
        'active_jobs': active_jobs,
        'completed_jobs': completed_jobs,
        'failed_jobs': failed_jobs,
        'system_status': 'operational'
    })


def process_file_async(job_id: str, file_path: str, api_key: str, model_type: str):
    """
    Process file asynchronously in background thread.
    
    Args:
        job_id (str): Unique job identifier
        file_path (str): Path to uploaded Excel file
        api_key (str): API key for LLM service
        model_type (str): Type of LLM model to use
    """
    job = processing_jobs.get(job_id)
    if not job:
        return
    
    try:
        logger.info(f"Starting processing job {job_id} with {model_type}")
        job.status = "processing"

        # Read products from Excel file and preserve original data
        product_info_list, original_data = ExcelHandler.read_input_file(file_path)
        job.total_products = len(product_info_list)
        job.original_data = original_data  # Store original data in job

        # Set batch size: 5 for >1000 products, else 3
        batch_size = 5 if len(product_info_list) > 1000 else 3
        llm_client = LLMClient(api_key, model_type)
        processor = BatchProcessor(llm_client, batch_size=batch_size)

        # Enhanced progress callback function that handles failed count
        def progress_callback(progress_pct: int, processed: int, total: int, failed: int = 0):
            # Only update progress and log after a batch is fully completed (to match frontend)
            if processed > 0:
                # Find the last product in the current batch
                current_product = None
                if len(processor.cache) > 0:
                    last_result = list(processor.cache.values())[-1]
                    current_product = last_result.get('product_name', 'Unknown')
                job.update_progress(processed, current_product or "")
                # Log progress for large datasets only after batch
                if total > 100 and processed % 50 == 0:
                    logger.info(f"Progress: {processed}/{total} ({progress_pct}%) processed, {failed} failed (batch complete)")

        # Stop check function
        def stop_check():
            return job.stop_requested

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                processor.process_products(product_info_list, progress_callback, stop_check)
            )
        finally:
            loop.close()

        # Helper to check which products are missing descriptions
        def get_missing_products(product_info_list, results):
            described_names = set()
            failed_products = []
            empty_descriptions = []
            
            for r in results:
                if isinstance(r, dict) and r.get('product_name'):
                    # Check if product has at least one description (short or long) AND is not explicitly failed
                    has_short = r.get('short_description', '').strip()
                    has_long = r.get('long_description', '').strip()
                    is_failed = r.get('status') == 'failed'
                    
                    if (has_short or has_long) and not is_failed:
                        described_names.add(r['product_name'])
                    elif is_failed:
                        failed_products.append(r['product_name'])
                    elif not has_short and not has_long:
                        empty_descriptions.append(r['product_name'])
            
            missing = [p for p in product_info_list if p.get('product_name') not in described_names]
            
            # Debug logging
            if failed_products:
                logger.debug(f"Products with failed status: {failed_products[:5]}")
            if empty_descriptions:
                logger.debug(f"Products with empty descriptions: {empty_descriptions[:5]}")
            if missing:
                missing_names = [p.get('product_name', 'Unknown') for p in missing[:5]]
                logger.debug(f"Products completely missing from results: {missing_names}")
            
            return missing

        # Handle processing results

        if not results:
            if job.stop_requested:
                raise Exception("Processing stopped by user request with no results")
            else:
                raise Exception("No results generated")

        # Filter out exceptions from failed tasks
        valid_results = []
        invalid_count = 0
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                invalid_count += 1
                logger.warning(f"Skipping invalid result: {result}")
        
        logger.info(f"Initial results: {len(results)} total, {len(valid_results)} valid, {invalid_count} invalid")
        
        # Debug: Check what we have in the valid results
        if valid_results:
            sample_result = valid_results[0]
            logger.debug(f"Sample result structure: {list(sample_result.keys())}")
            logger.debug(f"Sample result: {sample_result}")
        
        # Debug: Check initial missing count before retries
        initial_missing = get_missing_products(product_info_list, valid_results)
        logger.info(f"Before retries: {len(initial_missing)} products missing descriptions out of {len(product_info_list)} total")


        # Improved retry for missing products: always retry all missing in each round, robust merging, and log after each retry
        max_retries = 7
        retry_count = 0
        all_results = valid_results[:]
        previous_missing_count = -1
        stagnant_retries = 0
        
        while retry_count < max_retries:
            missing_products = get_missing_products(product_info_list, all_results)
            logger.warning(f"[Retry {retry_count+1}] {len(missing_products)} products missing descriptions.")
            
            # Check if we're making progress
            if len(missing_products) == previous_missing_count:
                stagnant_retries += 1
                logger.warning(f"No progress made in retry {retry_count+1} (stagnant count: {stagnant_retries})")
                if stagnant_retries >= 2:  # If no progress for 2 consecutive retries, stop
                    logger.error(f"Breaking retry loop: No progress after {stagnant_retries} consecutive attempts")
                    break
            else:
                stagnant_retries = 0  # Reset stagnant counter
                
            previous_missing_count = len(missing_products)
            
            # Log some missing product names for debugging
            if missing_products:
                missing_sample = [p.get('product_name', 'Unknown') for p in missing_products[:5]]
                logger.info(f"Sample missing products: {missing_sample}")
            
            if not missing_products or job.stop_requested:
                logger.info(f"✅ Retry loop completed: No more missing products after {retry_count} retries")
                break
                
            # Re-process all missing products
            logger.info(f"Retrying {len(missing_products)} missing products...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create a fresh processor for retries to avoid cache issues
                retry_processor = BatchProcessor(llm_client, batch_size=min(3, len(missing_products)))
                retry_results = loop.run_until_complete(
                    retry_processor.process_products(missing_products, progress_callback, stop_check)
                )
                logger.info(f"Retry produced {len(retry_results)} results")
                
                # Log details of retry results
                successful_retries = 0
                failed_retries = 0
                for r in retry_results:
                    if isinstance(r, dict):
                        if r.get('short_description', '').strip() or r.get('long_description', '').strip():
                            successful_retries += 1
                        else:
                            failed_retries += 1
                            logger.warning(f"Retry failed for {r.get('product_name', 'unknown')}: empty descriptions")
                    else:
                        failed_retries += 1
                        logger.warning(f"Invalid retry result: {r}")
                        
                logger.info(f"Retry breakdown: {successful_retries} successful, {failed_retries} failed")
                
            except Exception as e:
                logger.error(f"Retry processing failed: {str(e)}")
                retry_results = []
            finally:
                loop.close()
                
            # Debug: Check retry results
            valid_retry_results = [r for r in retry_results if isinstance(r, dict)]
            logger.info(f"Valid retry results: {len(valid_retry_results)}")
            
            # Merge: always keep only the latest result for each product_name, but prioritize results with descriptions
            result_map = {}
            
            # First, add all existing results
            for r in all_results:
                if isinstance(r, dict) and r.get('product_name'):
                    result_map[r['product_name']] = r
            
            logger.info(f"Current result_map has {len(result_map)} products")
            
            # Then, merge retry results - prioritize ones with actual descriptions
            retry_additions = 0
            retry_updates = 0
            for r in retry_results:
                if isinstance(r, dict) and r.get('product_name'):
                    product_name = r['product_name']
                    has_descriptions = r.get('short_description', '').strip() or r.get('long_description', '').strip()
                    
                    if product_name in result_map:
                        # Update existing result if the retry has descriptions or the existing one doesn't
                        existing_has_desc = (result_map[product_name].get('short_description', '').strip() or 
                                            result_map[product_name].get('long_description', '').strip())
                        if has_descriptions or not existing_has_desc:
                            result_map[product_name] = r
                            retry_updates += 1
                            if has_descriptions:
                                logger.debug(f"Updated {product_name} with retry result (has descriptions)")
                    else:
                        # New result
                        result_map[product_name] = r
                        retry_additions += 1
            
            logger.info(f"Retry merge: {retry_additions} new products, {retry_updates} updates")
            
            all_results = list(result_map.values())
            logger.info(f"After merge: {len(all_results)} total results")
            retry_count += 1

        # Targeted partial retries for missing short or long descriptions separately (batched by 10)
        product_meta_map = {p.get('product_name'): p for p in product_info_list if p.get('product_name')}

        def _classify_partial_results(results_list):
            missing_short = []
            missing_long = []
            for entry in results_list:
                if not isinstance(entry, dict):
                    continue
                pname = entry.get('product_name')
                if not pname:
                    continue
                short_desc = str(entry.get('short_description', '')).strip()
                long_desc = str(entry.get('long_description', '')).strip()
                status = entry.get('status')

                if short_desc and long_desc and status != 'failed':
                    continue

                if not short_desc:
                    missing_short.append(entry)
                if not long_desc:
                    missing_long.append(entry)

            return missing_short, missing_long

        async def _regenerate_descriptions_batched(entries, description_type: str, batch_size: int = 10):
            results = [None] * len(entries)
            async def _run_batch(batch_entries, start_index):
                tasks = []
                index_map = []
                for idx, item in enumerate(batch_entries):
                    pname = item.get('product_name')
                    if not pname:
                        continue
                    meta = product_meta_map.get(pname, {})
                    category = meta.get('category') if isinstance(meta, dict) else None
                    tasks.append(llm_client.generate_description(pname, description_type, category))
                    index_map.append(start_index + idx)
                if not tasks:
                    return
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, res in zip(index_map, batch_results):
                    results[idx] = res

            for start in range(0, len(entries), batch_size):
                batch_entries = entries[start:start + batch_size]
                await _run_batch(batch_entries, start)

            return results

        missing_short_entries, missing_long_entries = _classify_partial_results(all_results)

        if missing_short_entries:
            logger.info(f"Attempting targeted batch retry for {len(missing_short_entries)} products missing short descriptions")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                short_retry_results = loop.run_until_complete(_regenerate_descriptions_batched(missing_short_entries, 'short'))
            finally:
                loop.close()
            successes = 0
            failed = 0
            for entry, new_desc in zip(missing_short_entries, short_retry_results):
                pname = entry.get('product_name', 'Unknown')
                if isinstance(new_desc, Exception):
                    logger.debug(f"Short description batch retry failed for {pname}: {str(new_desc)}")
                    failed += 1
                    continue
                if new_desc is None or not str(new_desc).strip():
                    logger.debug(f"Short description batch retry returned empty result for {pname}")
                    failed += 1
                    continue
                cleaned = llm_client._clean_response(str(new_desc)) if hasattr(llm_client, '_clean_response') else str(new_desc)
                formatted = llm_client._format_short_description_safely(cleaned, pname) if hasattr(llm_client, '_format_short_description_safely') else cleaned.strip()
                if not formatted.strip():
                    logger.debug(f"Short description batch retry produced empty formatted output for {pname}")
                    failed += 1
                    continue
                
                # Update the entry (which is a reference to the dict in all_results)
                old_short = entry.get('short_description', '')
                entry['short_description'] = formatted.strip()
                successes += 1
                
                # Update status
                if entry.get('long_description', '').strip():
                    entry['status'] = 'success'
                else:
                    entry['status'] = 'partial'
                
                # Log successful update with length comparison
                logger.info(f"✓ Updated short description for '{pname[:40]}': {len(old_short)} → {len(formatted)} chars")
                
            logger.info(f"Short description batch retry: {successes} succeeded, {failed} failed out of {len(missing_short_entries)} total")

        if missing_long_entries:
            logger.info(f"Attempting targeted batch retry for {len(missing_long_entries)} products missing long descriptions")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                long_retry_results = loop.run_until_complete(_regenerate_descriptions_batched(missing_long_entries, 'long'))
            finally:
                loop.close()
            successes = 0
            failed = 0
            for entry, new_desc in zip(missing_long_entries, long_retry_results):
                pname = entry.get('product_name', 'Unknown')
                if isinstance(new_desc, Exception):
                    logger.debug(f"Long description batch retry failed for {pname}: {str(new_desc)}")
                    failed += 1
                    continue
                if new_desc is None or not str(new_desc).strip():
                    logger.debug(f"Long description batch retry returned empty result for {pname}")
                    failed += 1
                    continue
                cleaned = llm_client._clean_response(str(new_desc)) if hasattr(llm_client, '_clean_response') else str(new_desc)
                formatted = llm_client._format_long_description_safely(cleaned, pname) if hasattr(llm_client, '_format_long_description_safely') else cleaned.strip()
                if not formatted.strip():
                    logger.debug(f"Long description batch retry produced empty formatted output for {pname}")
                    failed += 1
                    continue
                
                # Update the entry (which is a reference to the dict in all_results)
                old_long = entry.get('long_description', '')
                entry['long_description'] = formatted.strip()
                successes += 1
                
                # Update status
                if entry.get('short_description', '').strip():
                    entry['status'] = 'success'
                else:
                    entry['status'] = 'partial'
                
                # Log successful update with length comparison
                logger.info(f"✓ Updated long description for '{pname[:40]}': {len(old_long)} → {len(formatted)} chars")
                
            logger.info(f"Long description batch retry: {successes} succeeded, {failed} failed out of {len(missing_long_entries)} total")

        # Final check for missing with comprehensive analysis
        missing_products = get_missing_products(product_info_list, all_results)
        if missing_products and not job.stop_requested:
            missing_names = [p.get('product_name', 'Unknown') for p in missing_products[:10]]  # Show first 10
            logger.error(f"After {max_retries} retries, {len(missing_products)} products still missing descriptions.")
            logger.error(f"First few missing products: {missing_names}")
            
            # Comprehensive analysis of why products are missing
            logger.info("=== MISSING PRODUCTS ANALYSIS ===")
            logger.info(f"Total input products: {len(product_info_list)}")
            logger.info(f"Total results generated: {len(all_results)}")
            
            # Check what happened to the first few missing products
            for i, missing_product in enumerate(missing_products[:5]):  # Check first 5 missing
                product_name = missing_product.get('product_name', '')
                logger.info(f"\n--- Missing Product #{i+1}: '{product_name}' ---")
                
                # Check if there's any result for this product name
                matching_results = [r for r in all_results if isinstance(r, dict) and r.get('product_name') == product_name]
                if matching_results:
                    result = matching_results[0]
                    short_desc = result.get('short_description', '').strip()
                    long_desc = result.get('long_description', '').strip()
                    status = result.get('status', 'unknown')
                    error = result.get('error', 'none')
                    
                    logger.info(f"  - Result exists with status: {status}")
                    logger.info(f"  - Short description length: {len(short_desc)}")
                    logger.info(f"  - Long description length: {len(long_desc)}")
                    logger.info(f"  - Error: {error}")
                    if short_desc:
                        logger.info(f"  - Short desc preview: '{short_desc[:50]}...'")
                    if long_desc:
                        logger.info(f"  - Long desc preview: '{long_desc[:50]}...'")
                else:
                    logger.info(f"  - NO RESULT FOUND for this product")
                    
                # Check for similar product names (fuzzy matching)
                similar_results = []
                product_name_lower = product_name.lower().replace(' ', '').replace('-', '').replace('_', '')
                for r in all_results:
                    if isinstance(r, dict) and r.get('product_name'):
                        r_name_lower = r['product_name'].lower().replace(' ', '').replace('-', '').replace('_', '')
                        if product_name_lower in r_name_lower or r_name_lower in product_name_lower:
                            similar_results.append(r['product_name'])
                            
                if similar_results:
                    logger.info(f"  - Similar names found: {similar_results[:3]}")
                else:
                    logger.info(f"  - No similar names found")
        else:
            logger.info(f"✅ All products processed successfully after {retry_count} retries.")

        # Store results in job for later download
        job.results = all_results

        # Create output file
        output_filename = f"output_{job_id}.xlsx"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Create the actual output file
        ExcelHandler.create_output_file(all_results, output_path, job.original_data)

        # Mark job as completed or stopped
        if job.stop_requested:
            job.stop(output_path)
            logger.info(f"Job {job_id} stopped by user. Processed {len(all_results)} products.")
        else:
            job.complete(output_path)
            logger.info(f"Job {job_id} completed successfully. Processed {len(all_results)} products.")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id} failed: {error_msg}")
        job.fail(error_msg)

    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error occurred.'}), 500


@app.route('/validate_api', methods=['POST'])
def validate_api():
    """
    Validate API key and check rate limits with real-time testing.
    Returns status, rate limit info, and validation results.
    """
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        model_type = data.get('model_type', 'mistral').lower()
        
        if not api_key:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'API key is required',
                'details': 'Please enter your API key'
            })
        
        openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
        allowed_models = openrouter_models + ['gemini']
        if model_type not in allowed_models:
            return jsonify({
                'success': False,
                'status': 'invalid',
                'message': 'Invalid model type',
                'details': 'Supported models: Mistral, OpenChat, DeepSeek, GPT-OSS (OpenRouter) and Gemini.'
            })
        
        # Test the API key with a simple request for supported models
        validation_result = test_api_key(api_key, model_type)
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error(f"API validation error: {str(e)}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': 'Validation failed',
            'details': str(e)
        })


def test_api_key(api_key: str, model_type: str) -> dict:
    """
    Test API key validity and check rate limits.
    
    Args:
        api_key (str): API key to test
        model_type (str): 'mistral' or 'gemini'
        
    Returns:
        dict: Validation results with status and rate limit info
    """
    import asyncio
    import time
    
    openrouter_models = ['mistral', 'openchat', 'deepseek', 'gptoss']
    async def async_test():
        try:
            # Initialize LLM client
            llm_client = LLMClient(api_key, model_type)
            # Test with a simple prompt
            test_prompt = "Describe the pharmaceutical uses of paracetamol."
            start_time = time.time()
            if model_type in openrouter_models:
                result = await llm_client._call_mistral(test_prompt, model_override=llm_client.openrouter_models[model_type], max_retries=1)
            else:  # gemini
                result = await llm_client._call_gemini(test_prompt, max_retries=1)
            response_time = time.time() - start_time
            if result and len(result.strip()) > 0:
                # API key is valid and working
                return {
                    'success': True,
                    'status': 'valid',
                    'message': f'{model_type.title()} API key is valid and working',
                    'details': {
                        'response_time': round(response_time, 2),
                        'model': model_type,
                        'rate_limit_status': 'Good',
                        'test_response_length': len(result),
                        'timestamp': time.strftime('%H:%M:%S')
                    }
                }
            else:
                # API responded but with empty result
                return {
                    'success': False,
                    'status': 'warning',
                    'message': f'{model_type.title()} API key responded but returned empty result',
                    'details': {
                        'response_time': round(response_time, 2),
                        'model': model_type,
                        'rate_limit_status': 'Unknown',
                        'suggestion': 'Check your API quota or try again'
                    }
                }
        except Exception as e:
            error_msg = str(e).lower()
            # Analyze the error type
            if 'unauthorized' in error_msg or 'invalid' in error_msg or '401' in error_msg:
                return {
                    'success': False,
                    'status': 'invalid',
                    'message': f'{model_type.title()} API key is invalid',
                    'details': {
                        'error': 'Authentication failed',
                        'suggestion': 'Check your API key is correct and active'
                    }
                }
            elif 'rate limit' in error_msg or '429' in error_msg:
                return {
                    'success': False,
                    'status': 'rate_limited',
                    'message': f'{model_type.title()} API key is valid but rate limited',
                    'details': {
                        'error': 'Rate limit exceeded',
                        'suggestion': 'Wait a few minutes and try again'
                    }
                }
            elif 'quota' in error_msg or 'billing' in error_msg or 'payment' in error_msg:
                return {
                    'success': False,
                    'status': 'quota_exceeded',
                    'message': f'{model_type.title()} API key quota exceeded',
                    'details': {
                        'error': 'Quota or billing issue',
                        'suggestion': 'Check your account billing and quota'
                    }
                }
            elif 'timeout' in error_msg or 'connection' in error_msg:
                return {
                    'success': False,
                    'status': 'connection_error',
                    'message': f'Connection error to {model_type.title()} API',
                    'details': {
                        'error': 'Network or timeout issue',
                        'suggestion': 'Check your internet connection'
                    }
                }
            else:
                return {
                    'success': False,
                    'status': 'error',
                    'message': f'{model_type.title()} API test failed',
                    'details': {
                        'error': str(e),
                        'suggestion': 'Check your API key and try again'
                    }
                }
    
    # Run the async test
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_test())
        loop.close()
        return result
    except Exception as e:
        return {
            'success': False,
            'status': 'error',
            'message': 'Validation system error',
            'details': {
                'error': str(e),
                'suggestion': 'Please try again'
            }
        }


def cleanup_old_jobs():
    """
    Clean up old processing jobs and files (run periodically).
    """
    cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hours ago
    
    jobs_to_remove = []
    for job_id, job in processing_jobs.items():
        if job.end_time and job.end_time < cutoff_time:
            # Clean up output file
            if job.output_file and os.path.exists(job.output_file):
                try:
                    os.remove(job.output_file)
                except Exception:
                    pass
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del processing_jobs[job_id]
    
    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


def start_cleanup_scheduler():
    """
    Start background thread for automatic file cleanup.
    Runs cleanup every 6 hours.
    """
    def cleanup_worker():
        while True:
            try:
                time.sleep(6 * 60 * 60)  # Sleep for 6 hours
                cleanup_old_jobs()
                logger.info("Automatic cleanup completed")
            except Exception as e:
                logger.error(f"Error in automatic cleanup: {str(e)}")
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Automatic cleanup scheduler started (runs every 6 hours)")


if __name__ == '__main__':
    logger.info("Starting Pharma Description Generator application...")
    
    # Show startup information
    print("\n" + "="*60)
    print("🧬 PHARMACEUTICAL DESCRIPTION GENERATOR")
    print("="*60)
    print("✅ Server starting on: http://127.0.0.1:5000")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("� Output: Direct download (no server storage)")
    print("📊 Max file size: 50MB")
    print("🤖 Supported models: Mistral 7B (OpenRouter), OpenChat, DeepSeek, GPT-OSS (OpenRouter) , Gemini 1.5 Flash")
    print("="*60)
    print("🚀 Open your browser and navigate to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Clean up any existing old files on startup
    cleanup_old_jobs()
    
    # Start automatic cleanup scheduler
    start_cleanup_scheduler()
    
    # Start the Flask development server
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
