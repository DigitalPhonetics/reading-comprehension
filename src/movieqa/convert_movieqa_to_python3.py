import os
import re


def read_lines(fname):
    with open(fname, "r") as f:
        return f.readlines()


def write_lines(lines, fname):
    with open(fname, "w") as f:
        for line in lines:
            f.write(line)


def convert_print_statements(py2_lines):
    py3_lines = []
    for py2_line in py2_lines:
        # print("py2 line %s" % py2_line)

        print_statement = re.match(r'(.*)print ["\'](.*)["\'](.*)', py2_line)

        if print_statement:
            # print("found print statement %s" % py2_line)
            indent = print_statement.group(1)
            statement = print_statement.group(2)
            print_args = print_statement.group(3)
            py3_line = "%sprint('%s')%s\n" % (indent, statement, print_args)
            # print("converted print statement to python 3 %s" % py3_line)
        else:
            py3_line = py2_line
        py3_lines.append(py3_line)

    return py3_lines


def convert_iteritems(py2_lines):
    py3_lines = []
    for py2_line in py2_lines:
        py3_line = py2_line.replace(".iteritems()", ".items()")
        py3_lines.append(py3_line)

    return py3_lines


def convert_movieqa_code():
    scripts = ['data/data_loader.py', 'data/story_loader.py']
    for script in scripts:
        py2_script = script + "2"
        os.rename(script, script + "2")

        py2_lines = read_lines(py2_script)
        py3_lines = convert_print_statements(py2_lines)
        py3_lines = convert_iteritems(py3_lines)

        write_lines(py3_lines, script)


if __name__ == "__main__":
    convert_movieqa_code()
