import sys
import struct
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QTreeWidget, QTreeWidgetItem, QFileDialog,
                           QLabel, QHBoxLayout, QMenuBar, QMenu, QMessageBox,
                           QSplitter, QFrame)
from PyQt6.QtCore import Qt
import json
import csv

class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

@dataclass
class GGUFHeader:
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int

class GGUFReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.header = self._read_header()
        self.metadata = self._read_metadata()
    
    def _read_header(self) -> GGUFHeader:
        magic = self.file.read(4)
        if magic != b'GGUF':
            raise ValueError("Invalid GGUF file (wrong magic)")
        version = struct.unpack('<I', self.file.read(4))[0]
        if version != 3:
            raise ValueError(f"Unsupported GGUF version: {version}")
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', self.file.read(8))[0]
        return GGUFHeader(magic, version, tensor_count, metadata_kv_count)

    def _read_string(self) -> str:
        length = struct.unpack('<Q', self.file.read(8))[0]
        return self.file.read(length).decode('utf-8')

    def _read_array(self) -> List[Any]:
        type_id = struct.unpack('<I', self.file.read(4))[0]
        length = struct.unpack('<Q', self.file.read(8))[0]
        array_type = GGUFValueType(type_id)
        return [self._read_value(array_type) for _ in range(length)]

    def _read_value(self, value_type: GGUFValueType) -> Any:
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('<B', self.file.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('<b', self.file.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', self.file.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', self.file.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack('<I', self.file.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack('<i', self.file.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack('<f', self.file.read(4))[0]
        elif value_type == GGUFValueType.BOOL:
            return bool(struct.unpack('<B', self.file.read(1))[0])
        elif value_type == GGUFValueType.STRING:
            return self._read_string()
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array()
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', self.file.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', self.file.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', self.file.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_metadata(self) -> Dict[str, Any]:
        metadata = {}
        for _ in range(self.header.metadata_kv_count):
            key = self._read_string()
            value_type = GGUFValueType(struct.unpack('<I', self.file.read(4))[0])
            value = self._read_value(value_type)
            metadata[key] = value
        return metadata

    def close(self):
        self.file.close()

class GGUFPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_metadata = None
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Top bar with file info
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        self.file_label = QLabel("No file loaded")
        self.load_button = QPushButton("Load GGUF File")
        self.load_button.clicked.connect(self.load_file)
        top_layout.addWidget(self.file_label)
        top_layout.addWidget(self.load_button)
        layout.addWidget(top_bar)
        
        # Tree widget for metadata
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setColumnWidth(0, 300)
        layout.addWidget(self.tree)
        
    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open GGUF File",
            "",
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if filename:
            try:
                self.file_label.setText(f"File: {filename}")
                reader = GGUFReader(filename)
                self.display_metadata(reader)
                reader.close()
            except Exception as e:
                self.file_label.setText(f"Error: {str(e)}")
                self.tree.clear()
                self.current_metadata = None

    def _add_metadata_item(self, parent: Optional[QTreeWidgetItem], key: str, value: Any):
        if isinstance(value, (list, tuple)):
            item = QTreeWidgetItem(parent or self.tree, [key, f"Array ({len(value)} items)"])
            for i, v in enumerate(value):
                self._add_metadata_item(item, f"[{i}]", v)
            return item
        elif isinstance(value, dict):
            item = QTreeWidgetItem(parent or self.tree, [key, f"Dictionary ({len(value)} items)"])
            for k, v in value.items():
                self._add_metadata_item(item, k, v)
            return item
        else:
            return QTreeWidgetItem(parent or self.tree, [key, str(value)])

    def display_metadata(self, reader: GGUFReader):
        self.tree.clear()
        
        # Store current metadata
        self.current_metadata = {
            "header": {
                "magic": reader.header.magic.decode(),
                "version": reader.header.version,
                "tensor_count": reader.header.tensor_count,
                "metadata_kv_count": reader.header.metadata_kv_count
            },
            "metadata": reader.metadata
        }
        
        # Add header information
        header_item = QTreeWidgetItem(self.tree, ["Header"])
        QTreeWidgetItem(header_item, ["Magic", self.current_metadata["header"]["magic"]])
        QTreeWidgetItem(header_item, ["Version", str(self.current_metadata["header"]["version"])])
        QTreeWidgetItem(header_item, ["Tensor Count", str(self.current_metadata["header"]["tensor_count"])])
        QTreeWidgetItem(header_item, ["Metadata KV Count", str(self.current_metadata["header"]["metadata_kv_count"])])
        
        # Add metadata
        metadata_item = QTreeWidgetItem(self.tree, ["Metadata"])
        for key, value in self.current_metadata["metadata"].items():
            self._add_metadata_item(metadata_item, key, value)
        
        self.tree.expandToDepth(1)

class GGUFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GGUF Metadata Viewer")
        self.setMinimumSize(1200, 800)
        
        # Create UI components first
        self.create_panels()
        self.create_menu_bar()
        self.setup_ui()
        
    def create_panels(self):
        self.left_panel = GGUFPanel()
        self.right_panel = GGUFPanel()
        self.comparison_tree = QTreeWidget()
        self.comparison_tree.setHeaderLabels(["Key", "Left Value", "Right Value"])
        self.comparison_tree.setColumnWidth(0, 300)
        self.comparison_tree.setColumnWidth(1, 300)
        self.comparison_tree.setVisible(False)

    def setup_ui(self):
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create splitter and add panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        layout.addWidget(self.comparison_tree)
        
        # Get references to the vertical scrollbars
        left_vsb = self.left_panel.tree.verticalScrollBar()
        right_vsb = self.right_panel.tree.verticalScrollBar()
       
        left_vsb.valueChanged.connect(right_vsb.setValue)
        right_vsb.valueChanged.connect(left_vsb.setValue)
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_left_action = file_menu.addAction("Open Left")
        open_left_action.triggered.connect(self.left_panel.load_file)
        
        open_right_action = file_menu.addAction("Open Right")
        open_right_action.triggered.connect(self.right_panel.load_file)
        
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        
        export_json = export_menu.addAction("Export as JSON")
        export_json.triggered.connect(self.export_json)
        
        export_csv = export_menu.addAction("Export as CSV")
        export_csv.triggered.connect(self.export_csv)
        
        export_txt = export_menu.addAction("Export as Text")
        export_txt.triggered.connect(self.export_txt)
        
        file_menu.addSeparator()
        
        # Compare action
        compare_action = file_menu.addAction("Compare")
        compare_action.triggered.connect(self.compare_metadata)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                if len(v) > 0:
                    items.append((new_key, ', '.join(str(x) for x in v)))
                else:
                    items.append((new_key, '[]'))
            else:
                items.append((new_key, v))
        return dict(items)

    def _compare_values(self, v1: Any, v2: Any) -> bool:
        if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            return len(v1) == len(v2) and all(self._compare_values(x, y) for x, y in zip(v1, v2))
        elif isinstance(v1, dict) and isinstance(v2, dict):
            return self._compare_dicts(v1, v2)
        else:
            return str(v1) == str(v2)

    def _compare_dicts(self, d1: Dict, d2: Dict) -> bool:
        if set(d1.keys()) != set(d2.keys()):
            return False
        return all(self._compare_values(d1[k], d2[k]) for k in d1.keys())

    def _add_comparison_item(self, parent: Optional[QTreeWidgetItem], key: str, 
                           left_value: Any, right_value: Any, different: bool = False):
        item = QTreeWidgetItem(parent or self.comparison_tree)
        item.setText(0, key)
        
        if isinstance(left_value, (dict, list, tuple)) and isinstance(right_value, (dict, list, tuple)):
            item.setText(1, f"{type(left_value).__name__} ({len(left_value)} items)")
            item.setText(2, f"{type(right_value).__name__} ({len(right_value)} items)")
            
            if isinstance(left_value, dict):
                all_keys = sorted(set(left_value.keys()) | set(right_value.keys()))
                for k in all_keys:
                    left_val = left_value.get(k, "N/A")
                    right_val = right_value.get(k, "N/A")
                    self._add_comparison_item(item, k, left_val, right_val, 
                                           not self._compare_values(left_val, right_val))
            else:
                for i in range(max(len(left_value), len(right_value))):
                    left_val = left_value[i] if i < len(left_value) else "N/A"
                    right_val = right_value[i] if i < len(right_value) else "N/A"
                    self._add_comparison_item(item, f"[{i}]", left_val, right_val,
                                           not self._compare_values(left_val, right_val))
        else:
            item.setText(1, str(left_value))
            item.setText(2, str(right_value))
        
        if different:
            for col in range(3):
                item.setBackground(col, Qt.GlobalColor.yellow)

    def compare_metadata(self):
        if not self.left_panel.current_metadata or not self.right_panel.current_metadata:
            QMessageBox.warning(self, "Error", "Please load both files before comparing.")
            return
        
        self.comparison_tree.clear()
        self.comparison_tree.setVisible(True)
        
        # Compare header information
        header_item = QTreeWidgetItem(self.comparison_tree, ["Header"])
        for key in self.left_panel.current_metadata["header"].keys():
            left_val = self.left_panel.current_metadata["header"][key]
            right_val = self.right_panel.current_metadata["header"][key]
            self._add_comparison_item(header_item, key, left_val, right_val,
                                   not self._compare_values(left_val, right_val))
        
        # Compare metadata
        metadata_item = QTreeWidgetItem(self.comparison_tree, ["Metadata"])
        all_keys = sorted(set(self.left_panel.current_metadata["metadata"].keys()) |
                        set(self.right_panel.current_metadata["metadata"].keys()))
        
        for key in all_keys:
            left_val = self.left_panel.current_metadata["metadata"].get(key, "N/A")
            right_val = self.right_panel.current_metadata["metadata"].get(key, "N/A")
            self._add_comparison_item(metadata_item, key, left_val, right_val,
                                   not self._compare_values(left_val, right_val))
        
        self.comparison_tree.expandToDepth(1)

    def get_export_data(self) -> Dict:
        # Gather comparison data if both panels have data
        if self.left_panel.current_metadata and self.right_panel.current_metadata:
            return {
                "left_file": self.left_panel.current_metadata,
                "right_file": self.right_panel.current_metadata,
                "differences": self._get_differences()
            }
        # Otherwise return data from whichever panel has data
        elif self.left_panel.current_metadata:
            return self.left_panel.current_metadata
        elif self.right_panel.current_metadata:
            return self.right_panel.current_metadata
        else:
            raise ValueError("No data to export")

    def _get_differences(self) -> Dict:
        if not (self.left_panel.current_metadata and self.right_panel.current_metadata):
            return {}
            
        differences = {"header": {}, "metadata": {}}
        
        # Compare header
        for key in self.left_panel.current_metadata["header"].keys():
            left_val = self.left_panel.current_metadata["header"][key]
            right_val = self.right_panel.current_metadata["header"][key]
            if not self._compare_values(left_val, right_val):
                differences["header"][key] = {"left": left_val, "right": right_val}
        
        # Compare metadata
        all_keys = set(self.left_panel.current_metadata["metadata"].keys()) | \
                  set(self.right_panel.current_metadata["metadata"].keys())
        
        for key in all_keys:
            left_val = self.left_panel.current_metadata["metadata"].get(key, "N/A")
            right_val = self.right_panel.current_metadata["metadata"].get(key, "N/A")
            if not self._compare_values(left_val, right_val):
                differences["metadata"][key] = {"left": left_val, "right": right_val}
        
        return differences

    def export_json(self):
        try:
            data = self.get_export_data()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
                QMessageBox.information(self, "Success", "Data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def export_csv(self):
        try:
            data = self.get_export_data()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            try:
                # Flatten the nested structure
                flattened_data = self._flatten_dict(data)
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Key', 'Value'])
                    for key, value in flattened_data.items():
                        writer.writerow([key, value])
                QMessageBox.information(self, "Success", "Data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

    def export_txt(self):
        try:
            data = self.get_export_data()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        if filename:
            try:
                flattened_data = self._flatten_dict(data)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    for key, value in flattened_data.items():
                        f.write(f"{key}: {value}\n")
                QMessageBox.information(self, "Success", "Data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")

def main():
    app = QApplication(sys.argv)
    viewer = GGUFViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()