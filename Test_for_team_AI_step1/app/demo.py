import os, typer
from dotenv import load_dotenv
from app.processors.video_processor import VideoProcessor

app = typer.Typer()

@app.command()
def main(input: str = typer.Option("input_videos", help="Folder or video file"),
         scene_threshold: int = typer.Option(int(os.getenv("SCENE_THRESHOLD","5")))):
    load_dotenv()
    proc = VideoProcessor()
    paths = []
    if os.path.isdir(input):
        for name in os.listdir(input):
            if name.lower().endswith((".mp4",".mov",".m4v",".mkv")):
                paths.append(os.path.join(input, name))
    elif os.path.isfile(input):
        paths = [input]
    else:
        typer.echo(f"No input found at {input}"); 
        raise typer.Exit(code=1)
    for p in paths[:10]:
        typer.echo(f"Processing: {p}")
        try:
            result = proc.process(p, scene_threshold=scene_threshold)
            typer.echo(f"Done: {result['filename']} → {len(result['highlights'])} highlights")
        except Exception as e:
            typer.echo(f"Error: {e}")

if __name__ == "__main__":
    app()
