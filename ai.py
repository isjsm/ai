#!/usr/bin/env python3
"""
AI Text Generator Pro - The Most Advanced Text Generation Tool for Linux
Version: 6.0 (Fixed & Enhanced)
"""

import sys
import os
import time
import logging
import json
import webbrowser
from datetime import datetime
from threading import Thread
import subprocess

# First, check for critical system dependencies
def check_system_dependencies():
    """Check for required system libraries before importing PyQt5"""
    try:
        # Check for libGL
        subprocess.run(["ldconfig", "-p", "|", "grep", "libGL.so.1"], 
                      shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        try:
            # Try to install libGL if missing
            print("🔧 Required system library libGL.so.1 not found. Attempting to install...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libgl1", "libglib2.0-0", "libxcb-xinerama0"], 
                          check=True)
            return True
        except Exception as e:
            print(f"❌ Failed to install system dependencies: {str(e)}")
            print("\nPlease install required system libraries manually:")
            print("sudo apt-get update")
            print("sudo apt-get install -y libgl1 libglib2.0-0 libxcb-xinerama0")
            return False

# Check system dependencies before importing PyQt5
if not check_system_dependencies():
    print("❌ Cannot proceed without required system libraries. Please install them and try again.")
    sys.exit(1)

# Now it's safe to import PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QTextEdit, QPushButton, QLabel, QProgressBar, QComboBox, 
        QSlider, QGroupBox, QTabWidget, QFileDialog, QCheckBox,
        QSplitter, QScrollArea, QFrame, QShortcut, QMenu, QMessageBox
    )
    from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QSettings
    from PyQt5.QtGui import (
        QFont, QColor, QPalette, QIcon, QKeySequence, 
        QTextCursor, QSyntaxHighlighter, QTextFormat
    )
    from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
except ImportError as e:
    print(f"❌ Critical error: Failed to import PyQt5: {str(e)}")
    print("\nPlease ensure system libraries are installed and try:")
    print("pip install PyQt5 --no-cache-dir")
    sys.exit(1)

# Import other dependencies with error handling
try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, pipeline,
        StoppingCriteria, StoppingCriteriaList
    )
    from tqdm import tqdm
    from colorama import init, Fore, Style
    from huggingface_hub import hf_hub_download
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from optimum.bettertransformer import BetterTransformer
    from qdarkstyle import load_stylesheet_pyqt5
except ImportError as e:
    print(f"❌ Missing required package: {str(e)}")
    print("\nPlease run: pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama
init(autoreset=True)

# Application settings
APP_NAME = "AI Text Generator Pro"
APP_VERSION = "6.0 (Fixed & Enhanced)"
APP_DESCRIPTION = "The most advanced AI text generation tool for Linux with comprehensive error handling"
APP_WEBSITE = "https://aitextgenpro.example.com"
APP_ICON = "resources/icon.png"

# Configure logging
log_file = os.path.expanduser('~/.ai_text_generator_pro.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file
)
logger = logging.getLogger("AI_Text_Generator_Pro")

class DependencyChecker:
    """Checks and verifies all required dependencies"""
    @staticmethod
    def check_pyqt5():
        """Verify PyQt5 is properly installed with all dependencies"""
        try:
            from PyQt5 import QtCore
            return True
        except ImportError as e:
            logger.error(f"PyQt5 import failed: {str(e)}")
            return False
    
    @staticmethod
    def check_torch():
        """Verify torch is properly installed"""
        try:
            import torch
            return torch.__version__
        except ImportError as e:
            logger.error(f"Torch import failed: {str(e)}")
            return None
    
    @staticmethod
    def check_cuda():
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

class ErrorHandler:
    """Handles errors with user-friendly messages and solutions"""
    @staticmethod
    def show_critical_error(parent, title, message, solution):
        """Show a critical error dialog with solution"""
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setInformativeText(f"Solution: {solution}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        logger.critical(f"Critical error: {title} - {message}\nSolution: {solution}")
    
    @staticmethod
    def show_warning(parent, title, message, solution=None):
        """Show a warning dialog"""
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        if solution:
            msg.setInformativeText(f"Solution: {solution}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        logger.warning(f"Warning: {title} - {message}\nSolution: {solution}")
    
    @staticmethod
    def handle_dependency_error(parent, error_type, error_details):
        """Handle dependency errors with specific solutions"""
        if "PyQt5" in error_type or "libGL" in error_type:
            ErrorHandler.show_critical_error(
                parent,
                "PyQt5 Dependency Error",
                f"Failed to initialize GUI system: {error_details}",
                "Please install required system libraries:\nsudo apt-get install -y libgl1 libglib2.0-0 libxcb-xinerama0"
            )
        elif "torch" in error_type.lower():
            ErrorHandler.show_critical_error(
                parent,
                "PyTorch Error",
                f"Failed to load PyTorch: {error_details}",
                "Try reinstalling PyTorch:\npip uninstall -y torch && pip install torch"
            )
        elif "model" in error_type.lower():
            ErrorHandler.show_warning(
                parent,
                "Model Loading Issue",
                f"Failed to load AI model: {error_details}",
                "Check your internet connection or try a different model"
            )
        else:
            ErrorHandler.show_critical_error(
                parent,
                "Application Error",
                f"An unexpected error occurred: {error_details}",
                "Please check the log file for details:\n" + log_file
            )

class SyntaxHighlighter(QSyntaxHighlighter):
    """Advanced syntax highlighting for generated text"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._formats = {}
        
        # Define formats for different text types
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(200, 100, 255))
        keyword_format.setFontWeight(QFont.Bold)
        self._formats['keyword'] = keyword_format
        
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(100, 200, 100))
        self._formats['string'] = string_format
        
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(120, 120, 120))
        comment_format.setFontItalic(True)
        self._formats['comment'] = comment_format
        
        self._rules = [
            (r'\b(if|else|for|while|def|class|return|import)\b', 'keyword'),
            (r'".*?"', 'string'),
            (r"#.*$", 'comment')
        ]

    def highlightBlock(self, text):
        for pattern, format_type in self._rules:
            import re
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, self._formats[format_type])

class ModelLoaderThread(QThread):
    """Thread for loading AI models without freezing the UI"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object)  # model, tokenizer
    error = pyqtSignal(str)

    def __init__(self, model_name, cache_dir, device, quantization):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.quantization = quantization
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.progress.emit(10, "Checking system resources...")
            
            # Check system resources with proper error handling
            try:
                has_gpu = torch.cuda.is_available()
                gpu_count = torch.cuda.device_count() if has_gpu else 0
                
                with open('/proc/meminfo', 'r') as f:
                    mem_total = int(f.readline().split()[1]) / 1024  # MB
            except Exception as e:
                logger.warning(f"Failed to check system resources: {str(e)}")
                has_gpu = False
                gpu_count = 0
                mem_total = 4096  # Assume 4GB if detection fails
            
            self.progress.emit(20, f"Detected {gpu_count} GPU(s) and {mem_total:.1f} MB RAM")
            
            # Create cache directory
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                self.progress.emit(30, f"Using cache directory: {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {str(e)}")
                self.error.emit(f"Failed to create cache directory: {str(e)}")
                return
            
            # Download model files with progress
            files = [
                "config.json",
                "generation_config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json"
            ]
            
            downloaded = 0
            total_files = len(files)
            
            for file in files:
                if not self._is_running:
                    return
                    
                try:
                    self.progress.emit(30 + int(50 * downloaded / total_files), 
                                      f"Downloading {file}...")
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=file,
                        local_dir=self.cache_dir,
                        local_dir_use_symlinks=False,
                        timeout=120  # Add timeout to prevent hanging
                    )
                    downloaded += 1
                except Exception as e:
                    logger.warning(f"Failed to download {file}: {str(e)}")
                    # Continue with other files instead of failing completely
            
            self.progress.emit(80, "Loading tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.cache_dir,
                    use_fast=True,
                    trust_remote_code=True,
                    local_files_only=True  # Use local files only after download
                )
                tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                logger.error(f"Tokenizer loading failed: {str(e)}")
                self.error.emit(f"Tokenizer loading failed: {str(e)}")
                return
            
            self.progress.emit(85, "Loading model with optimizations...")
            
            # Load model with appropriate quantization
            try:
                if self.quantization == "4-bit" and has_gpu:
                    from transformers import BitsAndBytesConfig
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        self.cache_dir,
                        quantization_config=nf4_config,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True
                    )
                elif self.quantization == "8-bit" and has_gpu:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.cache_dir,
                        load_in_8bit=True,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.cache_dir,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    
                    # Apply BetterTransformer for CPU optimization
                    try:
                        model = BetterTransformer.transform(model, keep_original_model=False)
                    except Exception as e:
                        logger.info(f"BetterTransformer not applicable: {str(e)}")
                
                self.progress.emit(95, "Finalizing model setup...")
                model.eval()
                
                if has_gpu:
                    torch.backends.cudnn.benchmark = True
                
                self.progress.emit(100, "Model loaded successfully!")
                time.sleep(0.5)  # Give UI time to update
                
                self.finished.emit(model, tokenizer)
                
            except Exception as e:
                logger.error(f"Model loading failed with local files: {str(e)}", exc_info=True)
                
                # Try without local_files_only as fallback
                try:
                    self.progress.emit(85, "Trying fallback model loading...")
                    
                    if self.quantization == "4-bit" and has_gpu:
                        from transformers import BitsAndBytesConfig
                        nf4_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            quantization_config=nf4_config,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    elif self.quantization == "8-bit" and has_gpu:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            load_in_8bit=True,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        
                        try:
                            model = BetterTransformer.transform(model, keep_original_model=False)
                        except:
                            pass
                    
                    model.eval()
                    self.progress.emit(100, "Model loaded successfully (fallback)!")
                    
                    self.finished.emit(model, tokenizer)
                except Exception as e2:
                    logger.error(f"Fallback model loading also failed: {str(e2)}", exc_info=True)
                    self.error.emit(f"Model loading failed: {str(e)}\nFallback also failed: {str(e2)}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}", exc_info=True)
            self.error.emit(f"Model loading failed: {str(e)}")

class TextGenerationThread(QThread):
    """Thread for text generation to keep UI responsive"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, float, int)  # generated_text, time, tokens
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt, params):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            start_time = time.time()
            
            # Tokenize input with error handling
            try:
                inputs = self.tokenizer(
                    self.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                    padding=True
                )
                
                # Move to proper device
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Input tokenization failed: {str(e)}")
                self.error.emit(f"Input processing failed: {str(e)}")
                return
            
            self.progress.emit(20)
            
            # Generate text
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.params['max_new_tokens'],
                        temperature=self.params['temperature'],
                        top_p=self.params['top_p'],
                        top_k=self.params['top_k'],
                        repetition_penalty=self.params['repetition_penalty'],
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
            except Exception as e:
                logger.error(f"Text generation failed: {str(e)}")
                self.error.emit(f"Text generation failed: {str(e)}")
                return
            
            self.progress.emit(80)
            
            # Decode output
            try:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt from generated text if it's included
                if self.prompt in generated_text:
                    generated_text = generated_text.replace(self.prompt, "", 1).strip()
            except Exception as e:
                logger.error(f"Text decoding failed: {str(e)}")
                self.error.emit(f"Text decoding failed: {str(e)}")
                return
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            tokens_generated = outputs.shape[1] if len(outputs.shape) > 1 else 0
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(generated_text, generation_time, tokens_generated)
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}", exc_info=True)
            self.error.emit(f"Text generation failed: {str(e)}")

class SettingsManager:
    """Manages application settings with validation"""
    def __init__(self):
        self.settings = QSettings("AI_Text_Generator_Pro", "Settings")
        self.validate_and_fix_settings()
    
    def validate_and_fix_settings(self):
        """Validate settings and fix any invalid values"""
        # Validate model settings
        model_name = self.settings.value('model/name', 'aubmindlab/ara-gpt2-base')
        if not model_name:
            self.settings.setValue('model/name', 'aubmindlab/ara-gpt2-base')
        
        cache_dir = self.settings.value('model/cache_dir', '')
        if not cache_dir or not os.path.isabs(cache_dir):
            self.settings.setValue('model/cache_dir', os.path.expanduser('~/.cache/ai_text_generator_pro/models'))
        
        device = self.settings.value('model/device', 'auto')
        if device not in ['auto', 'cpu', 'cuda']:
            self.settings.setValue('model/device', 'auto')
        
        quantization = self.settings.value('model/quantization', 'auto')
        if quantization not in ['auto', 'none', '4-bit', '8-bit']:
            self.settings.setValue('model/quantization', 'auto')
        
        # Validate generation settings
        max_tokens = self.settings.value('generation/max_tokens', 250, type=int)
        if max_tokens < 50 or max_tokens > 1000:
            self.settings.setValue('generation/max_tokens', 250)
        
        temperature = self.settings.value('generation/temperature', 0.6, type=float)
        if temperature < 0.1 or temperature > 1.5:
            self.settings.setValue('generation/temperature', 0.6)
        
        top_p = self.settings.value('generation/top_p', 0.85, type=float)
        if top_p < 0.1 or top_p > 1.0:
            self.settings.setValue('generation/top_p', 0.85)
        
        top_k = self.settings.value('generation/top_k', 40, type=int)
        if top_k < 5 or top_k > 200:
            self.settings.setValue('generation/top_k', 40)
        
        rep_penalty = self.settings.value('generation/rep_penalty', 1.2, type=float)
        if rep_penalty < 1.0 or rep_penalty > 2.0:
            self.settings.setValue('generation/rep_penalty', 1.2)
    
    def get(self, key, default=None):
        return self.settings.value(key, default)
    
    def set(self, key, value):
        self.settings.setValue(key, value)
        
    def get_model_settings(self):
        return {
            'model_name': self.get('model/name', 'aubmindlab/ara-gpt2-base'),
            'cache_dir': self.get('model/cache_dir', os.path.expanduser('~/.cache/ai_text_generator_pro/models')),
            'device': self.get('model/device', 'auto'),
            'quantization': self.get('model/quantization', 'auto')
        }
    
    def get_generation_settings(self):
        return {
            'max_new_tokens': self.get('generation/max_tokens', 250, type=int),
            'temperature': self.get('generation/temperature', 0.6, type=float),
            'top_p': self.get('generation/top_p', 0.85, type=float),
            'top_k': self.get('generation/top_k', 40, type=int),
            'repetition_penalty': self.get('generation/rep_penalty', 1.2, type=float)
        }
    
    def save_model_settings(self, settings):
        self.set('model/name', settings['model_name'])
        self.set('model/cache_dir', settings['cache_dir'])
        self.set('model/device', settings['device'])
        self.set('model/quantization', settings['quantization'])
    
    def save_generation_settings(self, settings):
        self.set('generation/max_tokens', settings['max_new_tokens'])
        self.set('generation/temperature', settings['temperature'])
        self.set('generation/top_p', settings['top_p'])
        self.set('generation/top_k', settings['top_k'])
        self.set('generation/rep_penalty', settings['repetition_penalty'])

class MainWindow(QMainWindow):
    """Main application window with comprehensive error handling"""
    def __init__(self):
        super().__init__()
        
        # Initialize settings
        self.settings_manager = SettingsManager()
        self.model = None
        self.tokenizer = None
        self.model_loader = None
        self.text_generator = None
        self.is_model_loaded = False
        self.is_initializing = True
        
        # Setup UI
        self.setWindowTitle(f"{APP_NAME} - {APP_VERSION}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create top control panel
        self.create_control_panel(main_layout)
        
        # Create main content area with tabs
        self.create_main_content(main_layout)
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing application...")
        
        # Load settings and initialize
        self.load_settings()
        self.initialize_application()
        
        # Apply dark theme
        self.setStyleSheet(load_stylesheet_pyqt5())
        
        # Show window
        self.show()
        
        # Final initialization step
        self.is_initializing = False
    
    def create_menu_bar(self):
        """Create application menu bar with error handling"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        new_action = file_menu.addAction("New")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_document)
        
        open_action = file_menu.addAction("Open")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_document)
        
        save_action = file_menu.addAction("Save")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_document)
        
        save_as_action = file_menu.addAction("Save As")
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_document_as)
        
        print_action = file_menu.addAction("Print")
        print_action.setShortcut("Ctrl+P")
        print_action.triggered.connect(self.print_document)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menu_bar.addMenu("Edit")
        
        undo_action = edit_menu.addAction("Undo")
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_text)
        
        redo_action = edit_menu.addAction("Redo")
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo_text)
        
        edit_menu.addSeparator()
        
        cut_action = edit_menu.addAction("Cut")
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self.cut_text)
        
        copy_action = edit_menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_text)
        
        paste_action = edit_menu.addAction("Paste")
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.paste_text)
        
        # Model menu
        model_menu = menu_bar.addMenu("Model")
        
        reload_action = model_menu.addAction("Reload Model")
        reload_action.triggered.connect(self.reload_model)
        
        settings_action = model_menu.addAction("Model Settings")
        settings_action.triggered.connect(self.show_model_settings)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        docs_action = help_menu.addAction("Documentation")
        docs_action.triggered.connect(lambda: webbrowser.open(APP_WEBSITE))
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about_dialog)
    
    def create_control_panel(self, layout):
        """Create the top control panel with buttons and settings"""
        control_panel = QGroupBox("AI Text Generation Controls")
        control_layout = QVBoxLayout(control_panel)
        
        # First row: Model status and control buttons
        top_row = QHBoxLayout()
        
        # Model status
        self.model_status = QLabel("Model Status: Not loaded")
        self.model_status.setStyleSheet("font-weight: bold; color: #FF5555;")
        top_row.addWidget(self.model_status)
        
        # Spacer
        top_row.addStretch()
        
        # Control buttons
        self.start_button = QPushButton("Start Generation")
        self.start_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_MediaPlay')))
        self.start_button.setFixedSize(150, 40)
        self.start_button.clicked.connect(self.start_generation)
        self.start_button.setEnabled(False)
        top_row.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_MediaStop')))
        self.stop_button.setFixedSize(100, 40)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)
        top_row.addWidget(self.stop_button)
        
        self.exit_button = QPushButton("Exit")
        self.exit_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_DialogCloseButton')))
        self.exit_button.setFixedSize(100, 40)
        self.exit_button.clicked.connect(self.close)
        top_row.addWidget(self.exit_button)
        
        control_layout.addLayout(top_row)
        
        # Second row: Generation parameters
        params_row = QHBoxLayout()
        
        # Max tokens
        tokens_layout = QVBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        self.tokens_slider = QSlider(Qt.Horizontal)
        self.tokens_slider.setRange(50, 1000)
        self.tokens_slider.setValue(250)
        self.tokens_slider.valueChanged.connect(self.update_tokens_label)
        self.tokens_value = QLabel("250")
        tokens_layout.addWidget(tokens_label)
        tokens_layout.addWidget(self.tokens_slider)
        tokens_layout.addWidget(self.tokens_value, 0, Qt.AlignCenter)
        params_row.addLayout(tokens_layout)
        
        # Temperature
        temp_layout = QVBoxLayout()
        temp_label = QLabel("Temperature:")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(1, 15)
        self.temp_slider.setValue(6)
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        self.temp_value = QLabel("0.6")
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_value, 0, Qt.AlignCenter)
        params_row.addLayout(temp_layout)
        
        # Top P
        top_p_layout = QVBoxLayout()
        top_p_label = QLabel("Top P:")
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setRange(10, 100)
        self.top_p_slider.setValue(85)
        self.top_p_slider.valueChanged.connect(self.update_top_p_label)
        self.top_p_value = QLabel("0.85")
        top_p_layout.addWidget(top_p_label)
        top_p_layout.addWidget(self.top_p_slider)
        top_p_layout.addWidget(self.top_p_value, 0, Qt.AlignCenter)
        params_row.addLayout(top_p_layout)
        
        # Top K
        top_k_layout = QVBoxLayout()
        top_k_label = QLabel("Top K:")
        self.top_k_slider = QSlider(Qt.Horizontal)
        self.top_k_slider.setRange(5, 200)
        self.top_k_slider.setValue(40)
        self.top_k_slider.valueChanged.connect(self.update_top_k_label)
        self.top_k_value = QLabel("40")
        top_k_layout.addWidget(top_k_label)
        top_k_layout.addWidget(self.top_k_slider)
        top_k_layout.addWidget(self.top_k_value, 0, Qt.AlignCenter)
        params_row.addLayout(top_k_layout)
        
        # Repetition Penalty
        rep_layout = QVBoxLayout()
        rep_label = QLabel("Rep Penalty:")
        self.rep_slider = QSlider(Qt.Horizontal)
        self.rep_slider.setRange(10, 20)
        self.rep_slider.setValue(12)
        self.rep_slider.valueChanged.connect(self.update_rep_label)
        self.rep_value = QLabel("1.2")
        rep_layout.addWidget(rep_label)
        rep_layout.addWidget(self.rep_slider)
        rep_layout.addWidget(self.rep_value, 0, Qt.AlignCenter)
        params_row.addLayout(rep_layout)
        
        control_layout.addLayout(params_row)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        layout.addWidget(control_panel)
    
    def create_main_content(self, layout):
        """Create the main content area with tabs"""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create input tab
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        
        # Prompt input
        prompt_label = QLabel("Enter your prompt:")
        input_layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Type your prompt here...")
        self.prompt_input.setFont(QFont("Arial", 12))
        self.prompt_input.setLineWrapMode(QTextEdit.WidgetWidth)
        input_layout.addWidget(self.prompt_input, 1)
        
        # Create output tab
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        # Generated text
        result_label = QLabel("Generated Text:")
        output_layout.addWidget(result_label)
        
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setFont(QFont("Arial", 12))
        self.result_output.setLineWrapMode(QTextEdit.WidgetWidth)
        output_layout.addWidget(self.result_output, 1)
        
        # Performance stats
        stats_layout = QHBoxLayout()
        self.time_label = QLabel("Time: 0.00s")
        self.tokens_label = QLabel("Tokens: 0")
        self.speed_label = QLabel("Speed: 0 tokens/s")
        stats_layout.addWidget(self.time_label)
        stats_layout.addWidget(self.tokens_label)
        stats_layout.addWidget(self.speed_label)
        stats_layout.addStretch()
        output_layout.addLayout(stats_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(input_tab, "Input")
        self.tab_widget.addTab(output_tab, "Output")
        
        layout.addWidget(self.tab_widget, 1)
    
    def load_settings(self):
        """Load application settings with validation"""
        try:
            # Load model settings
            model_settings = self.settings_manager.get_model_settings()
            
            # Load generation settings
            gen_settings = self.settings_manager.get_generation_settings()
            self.tokens_slider.setValue(gen_settings['max_new_tokens'])
            self.temp_slider.setValue(int(gen_settings['temperature'] * 10))
            self.top_p_slider.setValue(int(gen_settings['top_p'] * 100))
            self.top_k_slider.setValue(gen_settings['top_k'])
            self.rep_slider.setValue(int(gen_settings['repetition_penalty'] * 10))
            
            # Update labels
            self.update_tokens_label()
            self.update_temp_label()
            self.update_top_p_label()
            self.update_top_k_label()
            self.update_rep_label()
        except Exception as e:
            logger.error(f"Failed to load settings: {str(e)}")
            # Use default values if settings loading fails
            self.tokens_slider.setValue(250)
            self.temp_slider.setValue(6)
            self.top_p_slider.setValue(85)
            self.top_k_slider.setValue(40)
            self.rep_slider.setValue(12)
            
            self.update_tokens_label()
            self.update_temp_label()
            self.update_top_p_label()
            self.update_top_k_label()
            self.update_rep_label()
    
    def save_settings(self):
        """Save application settings with validation"""
        try:
            # Save model settings
            model_settings = {
                'model_name': "aubmindlab/ara-gpt2-base",
                'cache_dir': os.path.expanduser('~/.cache/ai_text_generator_pro/models'),
                'device': 'auto',
                'quantization': 'auto'
            }
            self.settings_manager.save_model_settings(model_settings)
            
            # Save generation settings
            gen_settings = {
                'max_new_tokens': self.tokens_slider.value(),
                'temperature': self.temp_slider.value() / 10.0,
                'top_p': self.top_p_slider.value() / 100.0,
                'top_k': self.top_k_slider.value(),
                'repetition_penalty': self.rep_slider.value() / 10.0
            }
            self.settings_manager.save_generation_settings(gen_settings)
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
    
    def initialize_application(self):
        """Initialize the application with comprehensive error handling"""
        try:
            self.status_bar.showMessage("Initializing application...")
            
            # Create cache directory
            cache_dir = os.path.expanduser('~/.cache/ai_text_generator_pro/models')
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Cache directory created at {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {str(e)}")
                ErrorHandler.show_warning(
                    self,
                    "Cache Directory Error",
                    "Failed to create cache directory",
                    f"Please check permissions for {cache_dir}"
                )
            
            # Check for critical dependencies
            if not DependencyChecker.check_pyqt5():
                ErrorHandler.show_critical_error(
                    self,
                    "PyQt5 Error",
                    "PyQt5 is not properly installed",
                    "Please run: pip install --upgrade PyQt5"
                )
                return
            
            torch_version = DependencyChecker.check_torch()
            if not torch_version:
                ErrorHandler.show_critical_error(
                    self,
                    "PyTorch Error",
                    "PyTorch is not installed or corrupted",
                    "Please run: pip install --upgrade torch"
                )
                return
            
            # Start loading model in background
            QTimer.singleShot(100, self.load_model)
            
        except Exception as e:
            logger.error(f"Application initialization failed: {str(e)}", exc_info=True)
            ErrorHandler.show_critical_error(
                self,
                "Initialization Error",
                f"Failed to initialize application: {str(e)}",
                "Please check the log file for details:\n" + log_file
            )
    
    def load_model(self):
        """Load the AI model in a background thread with fallbacks"""
        if self.is_initializing:
            return
            
        self.model_status.setText("Model Status: Loading...")
        self.model_status.setStyleSheet("font-weight: bold; color: #50C878;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Get model settings
        model_settings = self.settings_manager.get_model_settings()
        
        # Start model loader thread
        self.model_loader = ModelLoaderThread(
            model_name=model_settings['model_name'],
            cache_dir=model_settings['cache_dir'],
            device=model_settings['device'],
            quantization=model_settings['quantization']
        )
        self.model_loader.progress.connect(self.update_model_progress)
        self.model_loader.finished.connect(self.model_loaded)
        self.model_loader.error.connect(self.model_load_error)
        self.model_loader.start()
    
    def update_model_progress(self, value, message):
        """Update model loading progress"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(f"Loading model: {message}")
    
    def model_loaded(self, model, tokenizer):
        """Handle model loaded event"""
        self.model = model
        self.tokenizer = tokenizer
        self.is_model_loaded = True
        
        self.model_status.setText("Model Status: Ready")
        self.model_status.setStyleSheet("font-weight: bold; color: #50C878;")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        # Show GPU info if available
        gpu_info = ""
        if DependencyChecker.check_cuda():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            gpu_info = f" (GPU: {gpu_name})"
        
        self.status_bar.showMessage(f"Model loaded successfully!{gpu_info}")
        logger.info("AI model loaded successfully")
        
        # Save settings
        self.save_settings()
    
    def model_load_error(self, error_message):
        """Handle model loading error with user-friendly message"""
        self.model_status.setText("Model Status: Error")
        self.model_status.setStyleSheet("font-weight: bold; color: #FF5555;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        self.status_bar.showMessage(f"Error: {error_message}")
        logger.error(f"Model loading error: {error_message}")
        
        # Show error dialog with solutions
        ErrorHandler.show_warning(
            self,
            "Model Loading Error",
            "Failed to load AI model",
            error_message
        )
        
        # Offer solutions based on error type
        if "CUDA" in error_message or "GPU" in error_message:
            solution = ("Try running with CPU only:\n"
                       "1. Go to Model > Model Settings\n"
                       "2. Set Quantization to 'none'\n"
                       "3. Click OK and try reloading the model")
            ErrorHandler.show_warning(self, "GPU Issue", 
                                     "CUDA/GPU error detected", solution)
        elif "disk space" in error_message.lower() or "space" in error_message.lower():
            solution = ("Not enough disk space for model.\n"
                       "Please free up space or change cache directory in Model Settings.")
            ErrorHandler.show_warning(self, "Disk Space", 
                                     "Not enough disk space", solution)
        elif "permission" in error_message.lower():
            solution = ("Permission denied for cache directory.\n"
                       "Please check permissions for:\n" + 
                       self.settings_manager.get('model/cache_dir', ''))
            ErrorHandler.show_warning(self, "Permission Error", 
                                     "Access denied", solution)
    
    def start_generation(self):
        """Start text generation with input validation"""
        if self.is_initializing:
            return
            
        if not self.is_model_loaded:
            ErrorHandler.show_warning(
                self,
                "Model Not Loaded",
                "Cannot generate text",
                "The AI model is not loaded. Please wait for it to load or try reloading it."
            )
            return
        
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            ErrorHandler.show_warning(
                self,
                "Empty Prompt",
                "Please enter a prompt",
                "Type something in the input field before generating text."
            )
            return
        
        # Get generation parameters
        params = {
            'max_new_tokens': self.tokens_slider.value(),
            'temperature': self.temp_slider.value() / 10.0,
            'top_p': self.top_p_slider.value() / 100.0,
            'top_k': self.top_k_slider.value(),
            'repetition_penalty': self.rep_slider.value() / 10.0
        }
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Generating text...")
        
        # Start generation thread
        self.text_generator = TextGenerationThread(
            self.model, 
            self.tokenizer, 
            prompt, 
            params
        )
        self.text_generator.progress.connect(self.update_generation_progress)
        self.text_generator.finished.connect(self.generation_finished)
        self.text_generator.error.connect(self.generation_error)
        self.text_generator.start()
    
    def update_generation_progress(self, value):
        """Update text generation progress"""
        self.progress_bar.setValue(value)
    
    def generation_finished(self, generated_text, generation_time, tokens_generated):
        """Handle text generation finished event"""
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        
        # Display results
        if generated_text.strip():
            self.result_output.setPlainText(generated_text)
            self.tab_widget.setCurrentIndex(1)  # Switch to output tab
            
            # Update performance stats
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            self.time_label.setText(f"Time: {generation_time:.2f}s")
            self.tokens_label.setText(f"Tokens: {tokens_generated}")
            self.speed_label.setText(f"Speed: {tokens_per_second:.1f} tokens/s")
            
            self.status_bar.showMessage(f"Text generated successfully! ({generation_time:.2f}s)")
            logger.info(f"Text generated successfully in {generation_time:.2f}s")
        else:
            ErrorHandler.show_warning(
                self,
                "Empty Result",
                "Generated text is empty",
                "The model didn't produce any text. Try adjusting the parameters."
            )
    
    def generation_error(self, error_message):
        """Handle text generation error with specific solutions"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.status_bar.showMessage(f"Generation error: {error_message}")
        logger.error(f"Text generation error: {error_message}")
        
        # Show specific error with solutions
        ErrorHandler.show_warning(
            self, 
            "Generation Error", 
            "Failed to generate text",
            error_message
        )
        
        # Offer specific solutions based on error type
        if "CUDA" in error_message or "GPU" in error_message:
            solution = ("Try reducing the max tokens or switching to CPU mode:\n"
                       "1. Lower the Max Tokens slider\n"
                       "2. Go to Model > Model Settings\n"
                       "3. Set Quantization to 'none'\n"
                       "4. Click OK and reload the model")
            ErrorHandler.show_warning(self, "GPU Memory Error", 
                                     "CUDA out of memory", solution)
        elif "repetition" in error_message.lower():
            solution = ("Try lowering the Repetition Penalty value\n"
                       "or increasing the Temperature for more diversity.")
            ErrorHandler.show_warning(self, "Repetition Error", 
                                     "Repetition penalty issue", solution)
    
    def stop_generation(self):
        """Stop text generation safely"""
        if self.is_initializing:
            return
            
        if self.text_generator and self.text_generator.isRunning():
            self.text_generator.stop()
            self.text_generator.wait()
            self.text_generator = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Generation stopped by user")
            logger.info("Text generation stopped by user")
    
    def reload_model(self):
        """Reload the AI model with safety checks"""
        if self.is_initializing:
            return
            
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.stop()
            self.model_loader.wait()
        
        self.is_model_loaded = False
        self.model = None
        self.tokenizer = None
        
        # Reset UI state
        self.model_status.setText("Model Status: Reloading...")
        self.model_status.setStyleSheet("font-weight: bold; color: #FFA500;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        # Reload model after a short delay
        QTimer.singleShot(500, self.load_model)
    
    def update_tokens_label(self):
        """Update tokens slider label"""
        value = self.tokens_slider.value()
        self.tokens_value.setText(str(value))
    
    def update_temp_label(self):
        """Update temperature slider label"""
        value = self.temp_slider.value() / 10.0
        self.temp_value.setText(f"{value:.1f}")
    
    def update_top_p_label(self):
        """Update top-p slider label"""
        value = self.top_p_slider.value() / 100.0
        self.top_p_value.setText(f"{value:.2f}")
    
    def update_top_k_label(self):
        """Update top-k slider label"""
        value = self.top_k_slider.value()
        self.top_k_value.setText(str(value))
    
    def update_rep_label(self):
        """Update repetition penalty slider label"""
        value = self.rep_slider.value() / 10.0
        self.rep_value.setText(f"{value:.1f}")
    
    def new_document(self):
        """Create a new document"""
        if self.is_initializing:
            return
            
        self.prompt_input.clear()
        self.result_output.clear()
        self.tab_widget.setCurrentIndex(0)
    
    def open_document(self):
        """Open a document with error handling"""
        if self.is_initializing:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_input.setPlainText(content)
                self.tab_widget.setCurrentIndex(0)
                self.status_bar.showMessage(f"Opened: {file_path}")
                logger.info(f"Opened document: {file_path}")
            except Exception as e:
                logger.error(f"Failed to open file: {str(e)}")
                ErrorHandler.show_warning(
                    self, 
                    "Open Error", 
                    f"Failed to open file: {str(e)}",
                    "Check if the file exists and you have permission to read it."
                )
    
    def save_document(self):
        """Save the current document with error handling"""
        if self.is_initializing:
            return
            
        file_path = self.settings_manager.get("last_file_path", "")
        if not file_path:
            self.save_document_as()
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.prompt_input.toPlainText())
            self.status_bar.showMessage(f"Saved: {file_path}")
            logger.info(f"Document saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            ErrorHandler.show_warning(
                self, 
                "Save Error", 
                f"Failed to save file: {str(e)}",
                "Check if you have write permission for this location."
            )
    
    def save_document_as(self):
        """Save the current document with a new name"""
        if self.is_initializing:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Document", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.prompt_input.toPlainText())
                self.settings_manager.set("last_file_path", file_path)
                self.status_bar.showMessage(f"Saved: {file_path}")
                logger.info(f"Document saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save file: {str(e)}")
                ErrorHandler.show_warning(
                    self, 
                    "Save Error", 
                    f"Failed to save file: {str(e)}",
                    "Check if you have write permission for this location."
                )
    
    def print_document(self):
        """Print the current document with error handling"""
        if self.is_initializing:
            return
            
        printer = QPrinter(QPrinter.HighResolution)
        print_dialog = QPrintDialog(printer, self)
        
        if print_dialog.exec_() == QPrintDialog.Accepted:
            try:
                self.result_output.print_(printer)
                self.status_bar.showMessage("Document printed successfully")
                logger.info("Document printed")
            except Exception as e:
                logger.error(f"Print failed: {str(e)}")
                ErrorHandler.show_warning(
                    self, 
                    "Print Error", 
                    f"Failed to print: {str(e)}",
                    "Check your printer connection and settings."
                )
    
    def undo_text(self):
        """Undo text changes with context awareness"""
        if self.is_initializing:
            return
            
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.undo()
        else:
            self.result_output.undo()
    
    def redo_text(self):
        """Redo text changes with context awareness"""
        if self.is_initializing:
            return
            
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.redo()
        else:
            self.result_output.redo()
    
    def cut_text(self):
        """Cut selected text with context awareness"""
        if self.is_initializing:
            return
            
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.cut()
        else:
            self.result_output.cut()
    
    def copy_text(self):
        """Copy selected text with context awareness"""
        if self.is_initializing:
            return
            
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.copy()
        else:
            self.result_output.copy()
    
    def paste_text(self):
        """Paste text with context awareness"""
        if self.is_initializing:
            return
            
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.paste()
        else:
            self.result_output.paste()
    
    def show_model_settings(self):
        """Show model settings dialog with validation"""
        if self.is_initializing:
            return
            
        from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QComboBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Settings")
        dialog.resize(400, 200)
        
        layout = QFormLayout()
        
        # Model name (fixed for this version)
        model_name = QLineEdit("aubmindlab/ara-gpt2-base")
        model_name.setReadOnly(True)
        layout.addRow("Model Name:", model_name)
        
        # Cache directory
        cache_dir = QLineEdit(os.path.expanduser('~/.cache/ai_text_generator_pro/models'))
        cache_dir.setReadOnly(True)
        layout.addRow("Cache Directory:", cache_dir)
        
        # Device selection
        device_combo = QComboBox()
        device_combo.addItems(["Auto", "CPU", "CUDA"])
        current_device = self.settings_manager.get('model/device', 'auto').lower()
        if current_device == 'cuda':
            device_combo.setCurrentIndex(2)
        elif current_device == 'cpu':
            device_combo.setCurrentIndex(1)
        else:
            device_combo.setCurrentIndex(0)
        layout.addRow("Device:", device_combo)
        
        # Quantization selection
        quant_combo = QComboBox()
        quant_combo.addItems(["Auto", "None", "4-bit", "8-bit"])
        current_quant = self.settings_manager.get('model/quantization', 'auto').lower()
        if current_quant == '4-bit':
            quant_combo.setCurrentIndex(2)
        elif current_quant == '8-bit':
            quant_combo.setCurrentIndex(3)
        elif current_quant == 'none':
            quant_combo.setCurrentIndex(1)
        else:
            quant_combo.setCurrentIndex(0)
        layout.addRow("Quantization:", quant_combo)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # Save the selected settings
            device_map = {0: 'auto', 1: 'cpu', 2: 'cuda'}
            quant_map = {0: 'auto', 1: 'none', 2: '4-bit', 3: '8-bit'}
            
            new_settings = {
                'model_name': "aubmindlab/ara-gpt2-base",
                'cache_dir': os.path.expanduser('~/.cache/ai_text_generator_pro/models'),
                'device': device_map[device_combo.currentIndex()],
                'quantization': quant_map[quant_combo.currentIndex()]
            }
            
            self.settings_manager.save_model_settings(new_settings)
            self.reload_model()
    
    def show_about_dialog(self):
        """Show about dialog with system information"""
        if self.is_initializing:
            return
            
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea
        
        dialog = QDialog(self)
        dialog.setWindowTitle("About AI Text Generator Pro")
        dialog.resize(600, 500)
        
        main_layout = QVBoxLayout()
        
        # Logo and title
        title = QLabel(f"<h1>{APP_NAME}</h1>")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        version = QLabel(f"<h3>Version {APP_VERSION}</h3>")
        version.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(version)
        
        description = QLabel(APP_DESCRIPTION)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(description)
        
        # System information
        sys_info = QLabel(
            f"<b>System Information:</b><br/>"
            f"Python: {sys.version.split()[0]}<br/>"
            f"PyQt5: {'.'.join(map(str, PyQt5.QtCore.QT_VERSION_STR.split('.')[:3]))}<br/>"
            f"PyTorch: {DependencyChecker.check_torch() or 'Not available'}<br/>"
            f"CUDA: {'Available' if DependencyChecker.check_cuda() else 'Not available'}<br/>"
            f"Log file: {log_file}"
        )
        sys_info.setWordWrap(True)
        main_layout.addWidget(sys_info)
        
        # Features list
        features = QLabel(
            "<b>Key Features:</b><br/>"
            "- Comprehensive error handling and solutions<br/>"
            "- Automatic system dependency checks<br/>"
            "- Fallback mechanisms for model loading<br/>"
            "- User-friendly error messages with solutions<br/>"
            "- Detailed system information in About dialog<br/>"
            "- Input validation and parameter constraints<br/>"
            "- GPU/CPU auto-detection and optimization<br/>"
            "- Complete English interface"
        )
        features.setWordWrap(True)
        main_layout.addWidget(features)
        
        # Copyright
        copyright = QLabel(
            f"<br/>© {datetime.now().year} AI Text Generator Pro.<br/>"
            "All rights reserved.<br/>"
            f"<a href='{APP_WEBSITE}'>{APP_WEBSITE}</a>"
        )
        copyright.setAlignment(Qt.AlignCenter)
        copyright.setOpenExternalLinks(True)
        main_layout.addWidget(copyright)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        ok_button.setFixedSize(100, 30)
        main_layout.addWidget(ok_button, 0, Qt.AlignCenter)
        
        dialog.setLayout(main_layout)
        dialog.exec_()

    def closeEvent(self, event):
        """Handle application close event with cleanup"""
        # Stop any running threads
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.stop()
            self.model_loader.wait()
        
        if self.text_generator and self.text_generator.isRunning():
            self.text_generator.stop()
            self.text_generator.wait()
        
        # Save settings
        self.save_settings()
        
        logger.info("Application closed successfully")
        event.accept()

def main():
    """Main application entry point with comprehensive error handling"""
    # First, verify critical system dependencies
    if not check_system_dependencies():
        print("❌ Cannot proceed without required system libraries. Please install them and try again.")
        sys.exit(1)
    
    # Create application
    try:
        app = QApplication(sys.argv)
        
        # Set application attributes
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)
        app.setOrganizationName("AI Text Generator Pro")
        app.setOrganizationDomain("aitextgenpro.example.com")
        
        # Create and show main window
        window = MainWindow()
        
        # Start the application
        sys.exit(app.exec_())
        
    except Exception as e:
        # Critical error before UI can show
        print(f"❌ Critical error: {str(e)}")
        print("\nPlease check the log file for details:")
        print(log_file)
        logger.critical(f"Critical startup error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
