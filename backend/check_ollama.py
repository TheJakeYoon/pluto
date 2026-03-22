import ollama
import inspect

print("Methods in ollama:")
for name, obj in inspect.getmembers(ollama):
    if not name.startswith('_'):
        print(name)

print("\n---")
import pydoc
print(pydoc.render_doc(ollama.chat))
