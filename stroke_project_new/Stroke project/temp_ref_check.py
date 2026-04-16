import zipfile
import xml.etree.ElementTree as ET
import os
import re

def extract_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            content = z.read('word/document.xml')
        root = ET.fromstring(content)
        text = []
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        for para in root.findall('.//w:p', ns):
            p_text = "".join(node.text for node in para.findall('.//w:t', ns) if node.text)
            if p_text:
                text.append(p_text.strip())
        full_text = "\n".join(text)
        
        # Search for years 2023, 2024, 2025
        recent = re.findall(r'.*202[3-5].*', full_text)
        return "\n".join(recent)
    except Exception as e:
        return str(e)

path = r"C:\Users\Rekha\Desktop\stroke_project_new\stroke_project_new\Stroke project\Innovations_in_Stroke_Identification_A_Machine_Learning-Based_Diagnostic_Model_Using_Neuroimages.docx"
print(extract_text(path))
