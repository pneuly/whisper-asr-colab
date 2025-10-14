from whisper_asr_colab.docx_generator.docx_generator import DocxGenerator

def test_docx_generator(diarized_transcript):
        doc = DocxGenerator()
        doc.txt_to_word(diarized_transcript)
