import commands, yaml, sys
from time import sleep

def get_out_id(run_id):
    out = commands.getoutput("floyd info "+run_id)
    out = out.split("\n")
    for line in out:
        print line
        if "Output ID" in line:
            out_id = line.split(" ")[-1]

    return out_id

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



field = sys.argv[1]

if field != "test" and field != "train" and field != "data":
    print("Bad argument")
else:
    run_id = run_floyd(field)
    sleep(2)
    out_id = get_out_id(run_id)
    actualize_ids(field, run_id, out_id)
