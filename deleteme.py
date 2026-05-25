from PIL import Image, ImageFilter

source = Image.open(r"C:\Users\lupoleon\Desktop\1x\Artboard 1.png").convert("RGBA")
sizes = [16, 32, 48, 64, 128, 256]
frames = []

for s in sizes:
    frame = source.resize((s, s), Image.LANCZOS)
    if s <= 32:
        frame = frame.filter(ImageFilter.SHARPEN)
    frames.append(frame)


# Save: first image drives the file, the rest are appended
frames[0].save(
    "icon.ico",
    format="ICO",
    append_images=frames[1:],
    sizes=[(f.width, f.height) for f in frames]
)