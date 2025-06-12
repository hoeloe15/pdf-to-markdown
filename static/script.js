document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('uploadForm');
    const pdfFileInput = document.getElementById('pdfFile');
    const fileInputLabel = document.querySelector('.file-input-label');
    const fileInfo = document.getElementById('fileInfo');
    const convertBtn = document.getElementById('convertBtn');
    const backendSelect = document.getElementById('backend');
    const backendInfo = document.getElementById('backendInfo');
    const statusSection = document.getElementById('statusSection');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const successMessage = document.getElementById('successMessage');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    const downloadLink = document.getElementById('downloadLink');

    // File input change handler
    pdfFileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];

        if (file) {
            // Validate file type
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showError('Please select a PDF file.');
                resetFileInput();
                return;
            }

            // Validate file size (max 50MB)
            const maxSize = 50 * 1024 * 1024; // 50MB
            if (file.size > maxSize) {
                showError('File size must be less than 50MB.');
                resetFileInput();
                return;
            }

            // Update UI
            fileInputLabel.classList.add('has-file');
            fileInputLabel.querySelector('span').textContent = file.name;

            fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
            fileInfo.classList.add('show');

            convertBtn.disabled = false;
        } else {
            resetFileInput();
        }
    });

    // Backend selection change handler
    backendSelect.addEventListener('change', function (e) {
        const backend = e.target.value;

        if (backend === 'openai') {
            backendInfo.textContent = 'OpenAI processes PDFs directly for better accuracy';
        } else if (backend === 'azure') {
            backendInfo.textContent = 'Azure OpenAI converts PDF pages to images first';
        }
    });

    // Form submit handler
    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const file = pdfFileInput.files[0];
        if (!file) {
            showError('Please select a PDF file first.');
            return;
        }

        // Show loading state
        showLoading(file.size);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('backend', backendSelect.value);

            const response = await fetch('/api/convert', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showSuccess(result.download_url, result.filename, result.message, result.batched);
            } else {
                showError(result.detail || 'Conversion failed. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Network error. Please check your connection and try again.');
        }
    });

    function showLoading(fileSize) {
        statusSection.style.display = 'block';
        loadingIndicator.style.display = 'block';
        successMessage.style.display = 'none';
        errorMessage.style.display = 'none';

        // Update loading message based on file size
        const loadingText = document.querySelector('#loadingIndicator p');
        if (loadingText) {
            const fileSizeMB = fileSize / (1024 * 1024);
            if (fileSizeMB > 1 || fileSize > 10000) { // Rough estimate for larger files
                loadingText.innerHTML = `
                    <strong>🔄 Processing your PDF...</strong><br>
                    <span style="color: #666; font-size: 0.9em;">
                        Large file detected - using intelligent batching for optimal results
                    </span>
                `;
            } else {
                loadingText.innerHTML = '<strong>🔄 Converting PDF to Markdown...</strong>';
            }
        }

        // Disable form
        convertBtn.disabled = true;
        pdfFileInput.disabled = true;
    }

    function showSuccess(downloadUrl, filename, message, batched) {
        loadingIndicator.style.display = 'none';
        successMessage.style.display = 'block';
        errorMessage.style.display = 'none';

        downloadLink.href = downloadUrl;
        downloadLink.download = filename;
        downloadLink.textContent = `Download ${filename}`;

        // Show processing details
        const successText = document.querySelector('#successMessage p');
        if (successText) {
            if (batched) {
                // Extract batch info for a cleaner display
                const batchMatch = message.match(/processed in (\d+) batches due to size: ([^)]+)/);
                if (batchMatch) {
                    const [, numBatches, fileSize] = batchMatch;
                    successText.innerHTML = `
                        <strong>✅ Conversion completed successfully!</strong><br>
                        <span style="color: #666; font-size: 0.9em;">
                            📄 Large document processed in ${numBatches} parallel batches (${fileSize})<br>
                            🚀 Optimized for speed and quality
                        </span>
                    `;
                } else {
                    successText.innerHTML = `<strong>✅ ${message}</strong>`;
                }
            } else {
                successText.innerHTML = `<strong>✅ ${message}</strong>`;
            }
        }
    }

    function showError(message) {
        statusSection.style.display = 'block';
        loadingIndicator.style.display = 'none';
        successMessage.style.display = 'none';
        errorMessage.style.display = 'block';

        errorText.textContent = message;

        // Re-enable form
        convertBtn.disabled = pdfFileInput.files.length === 0;
        pdfFileInput.disabled = false;
    }

    function resetFileInput() {
        pdfFileInput.value = '';
        fileInputLabel.classList.remove('has-file');
        fileInputLabel.querySelector('span').textContent = 'Choose PDF File';
        fileInfo.classList.remove('show');
        convertBtn.disabled = true;
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Global function for reset button
    window.resetForm = function () {
        resetFileInput();
        statusSection.style.display = 'none';
        pdfFileInput.disabled = false;
    };

    // Drag and drop functionality
    const dropZone = document.querySelector('.file-input-label');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.style.borderColor = '#667eea';
        dropZone.style.backgroundColor = '#f0f4ff';
    }

    function unhighlight(e) {
        dropZone.style.borderColor = '#ddd';
        dropZone.style.backgroundColor = '#fafafa';
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            pdfFileInput.files = files;
            pdfFileInput.dispatchEvent(new Event('change'));
        }
    }
}); 