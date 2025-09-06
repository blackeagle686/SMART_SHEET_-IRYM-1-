document.addEventListener('DOMContentLoaded', function () {
    // ===== Sidebar toggle
    const sidebar_toggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    if (sidebar_toggle && sidebar) {
        sidebar_toggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }

    // ===== Summary Result Room - File Type Selector
const fileTypeSelect = document.getElementById('fileType');
const fileInput = document.getElementById('fileUpload');
const maxFileSizeMB = 50; // ✅ الحد الأقصى بالميجابايت

if (fileTypeSelect && fileInput) {
    fileTypeSelect.addEventListener('change', () => {
        const selectedType = fileTypeSelect.value;
        const fileTypes = {
            csv: '.csv',
            excel: '.xls,.xlsx',
            json: '.json',
            sql: '.sql',
        };
        fileInput.accept = fileTypes[selectedType] || '';
        fileInput.disabled = false;
        fileInput.value = ''; // إعادة تعيين أي ملف سابق
    });

    // ✅ تحقق من حجم الملف عند تغييره
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            const fileSizeMB = file.size / (1024 * 1024); // التحويل إلى MB
            if (fileSizeMB > maxFileSizeMB) {
                alert(`max size 50MG`);
                fileInput.value = ''; // إعادة تعيين الملف
            }
        }
    });
}


    // ===== Theme Handling
// ===== Theme Handling
const themeSelect = document.getElementById('themeSelect');
const body = document.body;
const savedTheme = localStorage.getItem('theme');

if (savedTheme) {
    body.setAttribute('data-theme', savedTheme);
    if (themeSelect) themeSelect.value = savedTheme;
} else {
    body.setAttribute('data-theme', 'light');
}

if (themeSelect) {
    themeSelect.addEventListener('change', function () {
        const selectedTheme = this.value;
        body.setAttribute('data-theme', selectedTheme);
        localStorage.setItem('theme', selectedTheme);
    });
}

    // ===== Upload file styling and preview
    const fileInputs = document.querySelectorAll('.upload-box input[type="file"]');
    const uploadSection = document.querySelector('.prompt-room-files-types-container');
    const previewSection = document.querySelector('.prompt-room-file-uploaded-preview');

    fileInputs.forEach(input => {
        input.addEventListener('change', function () {
            if (this.files && this.files.length > 0) {
                const fileName = this.files[0].name;
                const fileType = fileName.split('.').pop().toUpperCase();

                uploadSection.classList.add('fade-out');
                setTimeout(() => {
                    uploadSection.style.display = 'none';
                    previewSection.innerHTML = `
                        <div class="alert alert-info shadow fade-in" role="alert">
                            <h4 class="alert-heading"><i class="bi bi-file-earmark-check-fill"></i> File Uploaded</h4>
                            <p><strong>Type:</strong> ${fileType}</p>
                            <p><strong>Name:</strong> ${fileName}</p>
                            <hr>
                            <button class="btn btn-outline-primary btn-sm mt-2" id="reupload-btn">
                                <i class="bi bi-arrow-counterclockwise"></i> Re-upload
                            </button>
                        </div>
                    `;
                    previewSection.style.display = 'block';

                    document.getElementById('reupload-btn').addEventListener('click', () => {
                        uploadSection.style.display = 'flex';
                        previewSection.style.display = 'none';
                        uploadSection.classList.remove('fade-out');
                    });
                }, 500);
            }
        });
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const dash_td_all = document.querySelectorAll('.restlted-table td');

    function applyHeatmapColors() {
        const rootStyles = getComputedStyle(document.body);

        dash_td_all.forEach(cell => {
            const rawValue = cell.innerText.trim();
            const value = parseFloat(rawValue);

            if (isNaN(value)) return;

            let bgColor = '';
            let textColor = rootStyles.getPropertyValue('--light-color') || "#fff";

            if (value === 1) {
                bgColor = rootStyles.getPropertyValue('--primary-color');
            } else if (value >= 0.75) {
                bgColor = rootStyles.getPropertyValue('--accent-color');
            } else if (value >= 0.5) {
                bgColor = rootStyles.getPropertyValue('--mid-color');
                textColor = rootStyles.getPropertyValue('--dark-color');
            } else if (value > 0) {
                bgColor = rootStyles.getPropertyValue('--low-color');
                textColor = rootStyles.getPropertyValue('--dark-color');
            } else if (value === 0) {
                bgColor = rootStyles.getPropertyValue('--zero-color');
                textColor = rootStyles.getPropertyValue('--text-color');
            } else if (value > -0.5) {
                bgColor = rootStyles.getPropertyValue('--cool-color');
                textColor = rootStyles.getPropertyValue('--dark-color');
            } else if (value >= -0.75) {
                bgColor = rootStyles.getPropertyValue('--secondary-color');
            } else {
                bgColor = rootStyles.getPropertyValue('--secondary-color') || '#222';
            }

            cell.style.backgroundColor = bgColor;
            cell.style.color = textColor;
            cell.style.transition = 'all 0.3s ease';
        });
    }

    // ✅ أول تحميل للصفحة
    applyHeatmapColors();

    // ✅ إعادة التلوين عند تغيير الـ theme
    const themeSelect = document.getElementById('themeSelect');
    if (themeSelect) {
        themeSelect.addEventListener('change', () => {
            setTimeout(applyHeatmapColors, 50); // ندي وقت بسيط عشان تتطبق الـ CSS variables
        });
    }
});
