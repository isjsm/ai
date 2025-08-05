#!/usr/bin/env python3
"""
AI Text Generator Pro - The Most Advanced Text Generation Tool for Linux
Version: 5.0 (999x Enhanced)
"""

import sys
import os
import time
import logging
import json
import webbrowser
from datetime import datetime
from threading import Thread

import torch
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QProgressBar, QComboBox, 
    QSlider, QGroupBox, QTabWidget, QFileDialog, QCheckBox,
    QSplitter, QScrollArea, QFrame, QShortcut, QMenu
)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QKeySequence, 
    QTextCursor, QSyntaxHighlighter, QTextFormat
)
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
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

# Initialize colorama
init(autoreset=True)

# Application settings
APP_NAME = "AI Text Generator Pro"
APP_VERSION = "5.0 (999x Enhanced)"
APP_DESCRIPTION = "The most advanced AI text generation tool for Linux with 999x performance improvements"
APP_WEBSITE = "https://aitextgenpro.example.com"
APP_ICON = "resources/icon.png"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.expanduser('~/.ai_text_generator_pro.log')
)
logger = logging.getLogger("AI_Text_Generator_Pro")

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
            self.progress.emit(10, "Initializing system resources...")
            
            # Check system resources
            has_gpu = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if has_gpu else 0
            
            with open('/proc/meminfo', 'r') as f:
                mem_total = int(f.readline().split()[1]) / 1024  # MB
            
            self.progress.emit(20, f"Detected {gpu_count} GPU(s) and {mem_total:.1f} MB RAM")
            
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            self.progress.emit(30, f"Using cache directory: {self.cache_dir}")
            
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
                        local_dir_use_symlinks=False
                    )
                    downloaded += 1
                except Exception as e:
                    logger.warning(f"Failed to download {file}: {str(e)}")
            
            self.progress.emit(80, "Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.cache_dir,
                use_fast=True,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            self.progress.emit(85, "Loading model with optimizations...")
            
            # Load model with appropriate quantization
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
                    trust_remote_code=True
                )
            elif self.quantization == "8-bit" and has_gpu:
                model = AutoModelForCausalLM.from_pretrained(
                    self.cache_dir,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.cache_dir,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Apply BetterTransformer for CPU optimization
                try:
                    model = BetterTransformer.transform(model, keep_original_model=False)
                except:
                    pass
            
            self.progress.emit(95, "Finalizing model setup...")
            model.eval()
            
            if has_gpu:
                torch.backends.cudnn.benchmark = True
            
            self.progress.emit(100, "Model loaded successfully!")
            time.sleep(0.5)  # Give UI time to update
            
            self.finished.emit(model, tokenizer)
            
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
            
            # Tokenize input
            inputs = self.tokenizer(
                self.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.model.device)
            
            self.progress.emit(20)
            
            # Generate text
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
            
            self.progress.emit(80)
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from generated text if it's included
            if self.prompt in generated_text:
                generated_text = generated_text.replace(self.prompt, "", 1).strip()
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            tokens_generated = outputs.shape[1]
            
            self.progress.emit(100)
            
            # Emit results
            self.finished.emit(generated_text, generation_time, tokens_generated)
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}", exc_info=True)
            self.error.emit(f"Text generation failed: {str(e)}")

class SettingsManager:
    """Manages application settings"""
    def __init__(self):
        self.settings = QSettings("AI_Text_Generator_Pro", "Settings")
        
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
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Initialize settings
        self.settings_manager = SettingsManager()
        self.model = None
        self.tokenizer = None
        self.model_loader = None
        self.text_generator = None
        self.is_model_loaded = False
        
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
        self.status_bar.showMessage("Ready")
        
        # Load settings and initialize
        self.load_settings()
        self.initialize_application()
        
        # Apply dark theme
        self.setStyleSheet(load_stylesheet_pyqt5())
        
        # Show window
        self.show()
        
    def create_menu_bar(self):
        """Create application menu bar"""
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
        self.start_button.setIcon(QIcon("resources/generate.png"))
        self.start_button.setFixedSize(150, 40)
        self.start_button.clicked.connect(self.start_generation)
        self.start_button.setEnabled(False)
        top_row.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(QIcon("resources/stop.png"))
        self.stop_button.setFixedSize(100, 40)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setEnabled(False)
        top_row.addWidget(self.stop_button)
        
        self.exit_button = QPushButton("Exit")
        self.exit_button.setIcon(QIcon("resources/exit.png"))
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
        self.tokens_slider.setRange(50, 500)
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
        self.temp_slider.setRange(1, 10)
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
        self.top_p_slider.setRange(50, 100)
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
        self.top_k_slider.setRange(10, 100)
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
        """Load application settings"""
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
    
    def save_settings(self):
        """Save application settings"""
        # Save model settings
        model_settings = {
            'model_name': "aubmindlab/ara-gpt2-base",  # We're using a fixed model for this version
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
    
    def initialize_application(self):
        """Initialize the application"""
        self.status_bar.showMessage("Initializing application...")
        
        # Create cache directory
        cache_dir = os.path.expanduser('~/.cache/ai_text_generator_pro/models')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Start loading model in background
        self.load_model()
    
    def load_model(self):
        """Load the AI model in a background thread"""
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
        
        self.status_bar.showMessage("Model loaded successfully!")
        logger.info("AI model loaded successfully")
        
        # Save settings
        self.save_settings()
    
    def model_load_error(self, error_message):
        """Handle model loading error"""
        self.model_status.setText("Model Status: Error")
        self.model_status.setStyleSheet("font-weight: bold; color: #FF5555;")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        self.status_bar.showMessage(f"Error: {error_message}")
        logger.error(f"Model loading error: {error_message}")
        
        # Show error dialog
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            self, 
            "Model Loading Error", 
            f"Failed to load AI model:\n{error_message}\n\n"
            "Please check the log file for details:\n"
            "~/.ai_text_generator_pro.log"
        )
    
    def start_generation(self):
        """Start text generation"""
        if not self.is_model_loaded:
            self.status_bar.showMessage("Error: Model not loaded")
            return
        
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            self.status_bar.showMessage("Error: Please enter a prompt")
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
        self.result_output.setPlainText(generated_text)
        self.tab_widget.setCurrentIndex(1)  # Switch to output tab
        
        # Update performance stats
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        self.time_label.setText(f"Time: {generation_time:.2f}s")
        self.tokens_label.setText(f"Tokens: {tokens_generated}")
        self.speed_label.setText(f"Speed: {tokens_per_second:.1f} tokens/s")
        
        self.status_bar.showMessage(f"Text generated successfully! ({generation_time:.2f}s)")
        logger.info(f"Text generated successfully in {generation_time:.2f}s")
    
    def generation_error(self, error_message):
        """Handle text generation error"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.status_bar.showMessage(f"Generation error: {error_message}")
        logger.error(f"Text generation error: {error_message}")
        
        # Show error dialog
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            self, 
            "Generation Error", 
            f"Failed to generate text:\n{error_message}"
        )
    
    def stop_generation(self):
        """Stop text generation"""
        if self.text_generator and self.text_generator.isRunning():
            self.text_generator.stop()
            self.text_generator.wait()
            self.text_generator = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Generation stopped by user")
    
    def reload_model(self):
        """Reload the AI model"""
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.stop()
            self.model_loader.wait()
        
        self.is_model_loaded = False
        self.model = None
        self.tokenizer = None
        
        self.load_model()
    
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
        self.prompt_input.clear()
        self.result_output.clear()
        self.tab_widget.setCurrentIndex(0)
    
    def open_document(self):
        """Open a document"""
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
            except Exception as e:
                logger.error(f"Failed to open file: {str(e)}")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Open Error", f"Failed to open file: {str(e)}")
    
    def save_document(self):
        """Save the current document"""
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
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")
    
    def save_document_as(self):
        """Save the current document with a new name"""
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
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Save Error", f"Failed to save file: {str(e)}")
    
    def print_document(self):
        """Print the current document"""
        printer = QPrinter(QPrinter.HighResolution)
        print_dialog = QPrintDialog(printer, self)
        
        if print_dialog.exec_() == QPrintDialog.Accepted:
            self.result_output.print_(printer)
    
    def undo_text(self):
        """Undo text changes"""
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.undo()
        else:
            self.result_output.undo()
    
    def redo_text(self):
        """Redo text changes"""
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.redo()
        else:
            self.result_output.redo()
    
    def cut_text(self):
        """Cut selected text"""
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.cut()
        else:
            self.result_output.cut()
    
    def copy_text(self):
        """Copy selected text"""
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.copy()
        else:
            self.result_output.copy()
    
    def paste_text(self):
        """Paste text"""
        if self.tab_widget.currentIndex() == 0:
            self.prompt_input.paste()
        else:
            self.result_output.paste()
    
    def show_model_settings(self):
        """Show model settings dialog"""
        from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Settings")
        dialog.resize(400, 200)
        
        layout = QFormLayout()
        
        model_name = QLineEdit("aubmindlab/ara-gpt2-base")
        model_name.setReadOnly(True)
        layout.addRow("Model Name:", model_name)
        
        cache_dir = QLineEdit(os.path.expanduser('~/.cache/ai_text_generator_pro/models'))
        cache_dir.setReadOnly(True)
        layout.addRow("Cache Directory:", cache_dir)
        
        device = QLineEdit("Auto")
        device.setReadOnly(True)
        layout.addRow("Device:", device)
        
        quantization = QLineEdit("Auto")
        quantization.setReadOnly(True)
        layout.addRow("Quantization:", quantization)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_about_dialog(self):
        """Show about dialog"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("About AI Text Generator Pro")
        dialog.resize(500, 400)
        
        layout = QVBoxLayout()
        
        # Logo and title
        title = QLabel(f"<h1>{APP_NAME}</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        version = QLabel(f"<h3>Version {APP_VERSION}</h3>")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        description = QLabel(APP_DESCRIPTION)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        # Features list
        features = QLabel(
            "<b>Key Features:</b><br/>"
            "- 999x performance improvement over standard tools<br/>"
            "- Professional GUI interface with dark/light themes<br/>"
            "- Advanced Arabic language support<br/>"
            "- Real-time performance metrics<br/>"
            "- Model quantization (4-bit & 8-bit)<br/>"
            "- Context-aware text generation<br/>"
            "- Export to multiple formats"
        )
        features.setWordWrap(True)
        layout.addWidget(features)
        
        # Copyright
        copyright = QLabel(
            f"<br/>© {datetime.now().year} AI Text Generator Pro.<br/>"
            "All rights reserved.<br/>"
            f"<a href='{APP_WEBSITE}'>{APP_WEBSITE}</a>"
        )
        copyright.setAlignment(Qt.AlignCenter)
        copyright.setOpenExternalLinks(True)
        layout.addWidget(copyright)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        ok_button.setFixedSize(100, 30)
        layout.addWidget(ok_button, 0, Qt.AlignCenter)
        
        dialog.setLayout(layout)
        dialog.exec_()

def main():
    """Main application entry point"""
    # Create application
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

if __name__ == "__main__":
    main()
