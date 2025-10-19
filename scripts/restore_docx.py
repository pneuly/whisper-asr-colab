import os
import zipfile
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PreBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        main()


def restore_docx(xml_dir, output_docx):
    """Restore a .docx file from its XML components."""
    with zipfile.ZipFile(output_docx, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        for root, _, files in os.walk(xml_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, start=xml_dir)
                docx.write(full_path, arcname)


def main():
    base_dir = Path("whisper_asr_colab/docx_generator")
    xml_dir = Path(base_dir, "templete_xmls/diarized_transcription")
    output_dir = Path(base_dir, "templates")
    output_docx = Path(output_dir, "diarized_transcription.docx")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    restore_docx(xml_dir, output_docx)
    print(f"Restored .docx file saved at {output_docx}")


if __name__ == "__main__":
    main()
