import struct
import numpy as np
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
from collections import defaultdict

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

class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29

@dataclass
class GGUFHeader:
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int

@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: List[int]
    type: GGMLType
    offset: int

class GGUFUniqueValueCounter:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'rb')
        self.header = self._read_header()
        self.metadata = self._read_metadata()
        self.tensor_infos = self._read_tensor_infos()
        
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

    def _read_tensor_infos(self) -> List[TensorInfo]:
        tensor_infos = []
        for _ in range(self.header.tensor_count):
            name = self._read_string()
            n_dims = struct.unpack('<I', self.file.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack('<Q', self.file.read(8))[0])
            tensor_type = GGMLType(struct.unpack('<I', self.file.read(4))[0])
            offset = struct.unpack('<Q', self.file.read(8))[0]
            tensor_infos.append(TensorInfo(name, n_dims, dims, tensor_type, offset))
        return tensor_infos

    def get_tensor_data_offset(self) -> int:
        """ Calculate the offset where tensor data begins. """
        # Save current position
        current_pos = self.file.tell()
        
        # After header: go through metadata
        self.file.seek(24)  # Skip header
        for _ in range(self.header.metadata_kv_count):
            key = self._read_string()
            value_type = GGUFValueType(struct.unpack('<I', self.file.read(4))[0])
            _ = self._read_value(value_type)
        
        # After metadata: go through tensor infos
        for _ in range(self.header.tensor_count):
            name = self._read_string()
            n_dims = struct.unpack('<I', self.file.read(4))[0]
            for _ in range(n_dims):
                _ = struct.unpack('<Q', self.file.read(8))[0]
            _ = struct.unpack('<I', self.file.read(4))[0]  # type
            _ = struct.unpack('<Q', self.file.read(8))[0]  # offset
        
        # Align to 32 bytes
        pos = self.file.tell()
        alignment = 32
        aligned_pos = ((pos + alignment - 1) // alignment) * alignment
        
        # Restore position and return result
        self.file.seek(current_pos)
        return aligned_pos

    def dequantize_q4_0(self, data: bytes) -> np.ndarray:
        """ Dequantize Q4_0 format data """
        # Q4_0: Each block contains 1 scale (FP16) + 16 bytes of 4-bit quantized values (32 values)
        block_size = 18  # 2 bytes scale + 16 bytes data
        num_blocks = len(data) // block_size
        result = []
        
        for i in range(num_blocks):
            block_data = data[i * block_size:(i + 1) * block_size]
            # Read scale (FP16)
            scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
            
            # Read quantized values (4-bit each, 2 per byte)
            quant_values = []
            for j in range(16):
                byte = block_data[2 + j]
                low = byte & 0x0F
                high = (byte >> 4) & 0x0F
                quant_values.extend([low, high])
            
            # Dequantize
            for q in quant_values:
                # Q4_0 uses symmetric quantization [-8, 7]
                dequantized = (q - 8) * scale
                result.append(dequantized)
        
        return np.array(result, dtype=np.float32)

    def dequantize_q8_0(self, data: bytes) -> np.ndarray:
        """ Dequantize Q8_0 format data """
        # Q8_0: Each block contains 1 scale (FP16) + 32 bytes of 8-bit quantized values
        block_size = 34  # 2 bytes scale + 32 bytes data
        num_blocks = len(data) // block_size
        result = []
        
        for i in range(num_blocks):
            block_data = data[i * block_size:(i + 1) * block_size]
            # Read scale (FP16)
            scale = np.frombuffer(block_data[:2], dtype=np.float16)[0]
            
            # Read quantized values (8-bit each)
            quant_values = np.frombuffer(block_data[2:], dtype=np.int8)
            
            # Dequantize
            dequantized = quant_values.astype(np.float32) * scale
            result.extend(dequantized)
        
        return np.array(result, dtype=np.float32)

    def get_weight_values(self, tensor_info: TensorInfo) -> np.ndarray:
        """ Extract actual weight values, dequantizing if necessary """
        data_offset = self.get_tensor_data_offset()
        self.file.seek(data_offset + tensor_info.offset)
        
        # Calculate total elements
        total_elements = 1
        for dim in tensor_info.dims:
            total_elements *= dim
        
        if tensor_info.type == GGMLType.F32:
            raw_data = self.file.read(total_elements * 4)
            values = np.frombuffer(raw_data, dtype=np.float32)
        elif tensor_info.type == GGMLType.F16:
            raw_data = self.file.read(total_elements * 2)
            values = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
        elif tensor_info.type == GGMLType.Q4_0:
            # Calculate data size for Q4_0
            block_size = 32
            num_blocks = (total_elements + block_size - 1) // block_size
            raw_data = self.file.read(num_blocks * 18)  # 18 bytes per block
            values = self.dequantize_q4_0(raw_data)
        elif tensor_info.type == GGMLType.Q8_0:
            # Calculate data size for Q8_0
            block_size = 32
            num_blocks = (total_elements + block_size - 1) // block_size
            raw_data = self.file.read(num_blocks * 34)  # 34 bytes per block
            values = self.dequantize_q8_0(raw_data)
        else:
            # For other quantization types, return None
            return None
        
        return values

    def analyze_unique_values(self):
        """ Analyze unique values across all tensors """
        print(f"\nAnalyzing unique weight values in {self.filename}")
        print("=" * 80)
        
        all_unique_values = set()
        value_counts = defaultdict(int)
        tensor_unique_counts = {}
        total_values = 0
        supported_tensors = 0
        
        for i, tensor_info in enumerate(self.tensor_infos):
            print(f"Processing tensor {i+1}/{self.header.tensor_count}: {tensor_info.name}")
            
            try:
                values = self.get_weight_values(tensor_info)
                if values is not None:
                    supported_tensors += 1
                    flat_values = values.flatten()
                    
                    # Count unique values in this tensor
                    unique_in_tensor = np.unique(flat_values)
                    tensor_unique_counts[tensor_info.name] = len(unique_in_tensor)
                    
                    # Add to global unique set and count occurrences
                    for val in flat_values:
                        # Round to avoid floating point comparison issues
                        rounded_val = round(float(val), 6)
                        all_unique_values.add(rounded_val)
                        value_counts[rounded_val] += 1
                    
                    total_values += len(flat_values)
                    
                    print(f"  Unique values in this tensor: {len(unique_in_tensor):,}")
                    print(f"  Total values in this tensor: {len(flat_values):,}")
                    print(f"  Uniqueness ratio: {len(unique_in_tensor) / len(flat_values) * 100:.2f}%")
                else:
                    print(f"  Skipping (unsupported type: {tensor_info.type.name})")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Print overall analysis
        print("\n" + "=" * 80)
        print("OVERALL ANALYSIS")
        print("=" * 80)
        print(f"Total tensors analyzed: {supported_tensors}/{self.header.tensor_count}")
        print(f"Total weight values: {total_values:,}")
        print(f"Total unique values: {len(all_unique_values):,}")
        print(f"Overall uniqueness ratio: {len(all_unique_values) / total_values * 100:.2f}%")
        
        # Find most common values
        sorted_value_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nMost common values:")
        for val, count in sorted_value_counts[:10]:
            percentage = count / total_values * 100
            print(f"  {val:10.6f}: {count:,} occurrences ({percentage:.2f}%)")
        
        # Find tensors with lowest uniqueness
        sorted_tensors = sorted(tensor_unique_counts.items(), key=lambda x: x[1])
        print("\nTensors with fewest unique values:")
        for name, count in sorted_tensors[:10]:
            print(f"  {name}: {count:,} unique values")
        
        # Distribution analysis
        unique_counts = list(tensor_unique_counts.values())
        if unique_counts:
            print("\nUnique value distribution across tensors:")
            print(f"  Min unique values: {min(unique_counts):,}")
            print(f"  Max unique values: {max(unique_counts):,}")
            print(f"  Mean unique values: {np.mean(unique_counts):.2f}")
            print(f"  Median unique values: {np.median(unique_counts):.2f}")

    def close(self):
        self.file.close()

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_gguf_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    counter = GGUFUniqueValueCounter(filename)
    counter.analyze_unique_values()
    counter.close()