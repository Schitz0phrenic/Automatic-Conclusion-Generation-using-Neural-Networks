import json
from typing import Callable, Any


def __is_int__(value: str):
    """
    Checks if value is int-parsable.

    :param value: any str
    :return: bool
    """
    try:
        int(value)
        return True
    except ValueError:
        return False


def __is_float__(value: str):
    """
        Checks if value is float-parsable.

        :param value: any str
        :return: bool
        """
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_list_dict(x: str):
    """
    If x is given as a json-str defining a list or dict, it will be returned as such.

    :param x: any str
    :return: list or dict
    """
    return json.loads(x)


class IniReader:
    PARSERS = {str: lambda x: x, int: lambda x: int(x), float: lambda x: float(x), dict: parse_list_dict,
               list: parse_list_dict, bool: lambda x: x.lower() in ["true", "1"]}  # Dictionary containing all parser known to this class

    def __init__(self, file_path: str):
        """
        Creates a new IniReader that reads from the given file.

        :param file_path:
        """
        self.file_path = file_path
        self.sections = {"\n": {}}
        self.__initial_creation__()

    def contains_key(self, key: str, section="\n"):
        """
        checks if the ini file contained an entry.

        :param key: name of the entry
        :param section: section in which it is found
        :return:
        """
        return self.contains_section(section) and key in self.sections[section]

    def contains_section(self, section: str):
        """
        Checks if the ini contained a certain section

        :param section: name of the section
        :return: bool
        """
        return section in self.sections

    def get_item(self, key: str, section="\n", return_type: type = str):
        """
        Retrieves an item from the ini.

        :param key: name of the item
        :param section: section in which it is found. Use '\n' if no section is defined
        :param return_type: If there is a parser registered for this type, it will automatically try to parse the value.
        :return: the value.
        """
        parsers = type(self).PARSERS
        if return_type not in parsers:
            raise NotImplementedError(f"There is no parser implemented for type '{return_type}'!")
        return parsers[return_type](self[section][key])
    
    def set_item(self, data, key: str, section="\n"):
        """
        Adds another entry to the settings file.
        
        :param data: value that shall be written.
        :param key: which key shall be (over)written.
        :param section: In which section shall it be saved.
        """
        if section not in self.sections:
            self.sections[section] = {}
        self.sections[section][key] = data
    
    def write(self, filename=None):
        """
        Writes the current ini file to file.
        
        :param filename: If set, it will be written to this path, otherwise it will be written to the 
                         source file.
        """
        if not filename:
            filename = self.file_path
        with open(filename, "w") as f:
            default = self.sections["\n"]
            for k in default:
                val = str(default[k]).replace('\\', '\\\\').replace('\n', '\\n')
                f.write(f"{k} = {val}\n")
            f.write("\n")
            for section in self.sections:
                if section == "\n":
                    continue
                f.write(f"[{section}]\n")
                section = self.sections[section]
                for k in section:
                    val = str(section[k]).replace("\\", "\\\\").replace("\n", "\\n")
                    f.write(f'{k} = {val}\n')
                f.write("\n")

    @classmethod
    def add_parser(cls, parser: Callable[[str], Any], return_type: type):
        """
        Here you can add new parsers for the get_item method.

        :param parser: Method that parses a str to the return_type.
        :param return_type: Type which will be parsed into.
        :return: None
        """
        cls.PARSERS[return_type] = parser

    def __getitem__(self, item):
        """
        Returns all key->value pairs of a section.
        :param item: Section name
        :return: dict
        """
        if not self.contains_section(item):
            raise ValueError("Section '{}' does not exist!".format(item))
        return self.sections[item]

    def __initial_creation__(self):
        with open(self.file_path, "r") as f:
            content = f.readlines()
            f.close()
            current_section = "\n"
            current_line = 0
            max_line = len(content)
            while current_line < max_line:
                line = content[current_line].strip()
                current_line += 1
                if ";" in line:
                    line = line.split(";", maxsplit=1)[0]
                if len(line) == 0:
                    continue
                elif line[0] == "[":
                    if not line[-1] == "]":
                        raise SyntaxError("Syntax error in file '{}' line {}.".format(self.file_path, current_line))
                    current_section = line[1:-1]
                    if current_section not in self.sections:
                        self.sections[current_section] = {}
                elif "=" in line:
                    key, value = line.split("=", maxsplit=1)
                    key = key.strip()
                    value = value.strip()
                    value = value.replace("\\n", "\n").replace("\\\\", "\\")
                    if len(value) == 0:
                        raise SyntaxError("Syntax error in file '{}' line {}. Missing value!".format(self.file_path,
                                                                                                     current_line))
                    self.sections[current_section][key] = value
                else:
                    raise SyntaxError("Syntax error in file '{}' line {}. "
                                      "Line is no value, comment or section-definition".format(self.file_path,
                                                                                               current_line))
