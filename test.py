import typer

app = typer.Typer()


@app.command("prepare")
def prepare(a:str = typer.Option(..., "--data_path")):
    print(a)

@app.command("do")
def prepare(a:str = typer.Option(..., "--data_path")):
    print(a+"haha")

if __name__ == "__main__":
    app()

