import commands, yaml, sys, subprocess
from time import sleep
import click

def get_out_id(run_id):
    out = commands.getoutput("floyd info "+run_id)
    out = out.split("\n")
    for line in out:
        print line
        if "Output ID" in line:
            out_id = line.split(" ")[-1]
            return out_id
    raise ValueError("Output ID not found")

def get_run_id(field):
    file_name = "floyd/ids.yml"
    with open(file_name, 'r') as stream:
        out = yaml.load(stream)
        run_id = out[field]["run"]
    return run_id


def actualize_ids(field, run_id, out_id):
    file_name = "floyd/ids.yml"
    with open(file_name, 'r') as stream:
        out = yaml.load(stream)
        out[field]["run"] = run_id
        out[field]["output"] = out_id
        print out[field]
    with open(file_name, 'w') as outfile:
        yaml.dump(out, outfile, default_flow_style=False)

def run_floyd(script):
    out = commands.getoutput("floyd/"+script)
    out = out.split("\n")
    for i in range(len(out)):
        print out[i]
        if "RUN ID" in out[i]:
            out_id = out[i+2].split(" ")[0]
            return out_id
    raise ValueError("RUN ID not found")

def run_floyd_command(command, opt, id_):
    subprocess.call( " ".join(["floyd ",command,opt,id_]), shell=True)


@click.command()
@click.argument('script')
@click.argument('command', required=False)
@click.argument('opt', required=False)
def main(script, command=None, opt=None):
    if script != "test" and script != "train" and script != "data":
        raise ValueError("Bad argument")
    else:
        if command ==  None:
            run_id = run_floyd(script)
            sleep(2)
            out_id = get_out_id(run_id)
            actualize_ids(script, run_id, out_id)
        else:
            opt = "-"+opt if opt!=None else ""
            run_id = get_run_id(script)
            run_floyd_command(command, opt, run_id)

if __name__ == '__main__':
    main()
