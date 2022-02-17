#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import re
import csv


def get_attribute(node, allowed_keys, toupper=True):
    if node is None:
        return None
    for key in allowed_keys:
        if key in node.attrib:
            a = node.attrib[key]
            if toupper:
                return a.upper()
            else:
                return a
    return None


def get_first_child(node, possible_tags):
    if node is None:
        return None
    for child in node:
        for tag in possible_tags:
            if child.tag == tag:
                return child
    return None


def get_node(root, search_key, parent_key, parent=None):
    if parent is None:
        parent = root
    for child in parent:
        if child.tag not in ("s", "setting", "group"):
            continue
        key = child.attrib.get("key")
        if not key:
            key = child.attrib.get("k")
        if not key:
            continue

        # Check if keys match.
        current_key = parent_key + key
        if search_key == current_key:
            return child

        # Otherwise keep searching.
        next_child = get_node(root, search_key, current_key + "/", child)
        if next_child is not None:
            return next_child
    return None


def get_node_text(node):
    if node.text is None:
        return None
    pattern = re.compile(r"\s+")
    txt = node.text
    txt = re.sub(pattern, " ", txt)
    return txt.strip()


def get_type_params(node):
    # TODO Ignore csv inside quotes
    # http://stackoverflow.com/questions/8208358/split-string-ignoring-delimiter-within-quotation-marks-python
    params = get_node_text(node)
    if params is None:
        return None
    if '"' in params:
        x = csv.reader([params], skipinitialspace=True)
        params = next(x)
    else:
        params = params.split(",")
    return params


def format_double_string(string):
    if "e" not in string:
        value = string.split(".")
        if len(value) == 1:
            return "%.1f" % float(string)
        else:
            return "%.*f" % (len(value[1]), float(string))


def get_description(node, indent):
    for child in node:
        if child.tag == "desc" or child.tag == "description":
            # Extract the contents of the description tag as a string.
            desc = ET.tostring(child, method="xml").decode()
            desc = desc.replace("<desc>", "")
            desc = desc.replace("</desc>", "")
            desc = desc.replace("<description>", "")
            desc = desc.replace("</description>", "")

            # Replace any XML formatting with reStructuredText formatting.
            pattern = re.compile(r"\s+")
            desc = re.sub(pattern, " ", desc)
            desc = desc.replace("&amp;#8209;", "-")
            desc = desc.replace("<b><code>", "**")
            desc = desc.replace("<code><b>", "**")
            desc = desc.replace(" </b></code>", "**")
            desc = desc.replace(" </code></b>", "**")
            desc = desc.replace("</b></code>", "**")
            desc = desc.replace("</code></b>", "**")
            desc = desc.replace("<b>", "**")
            desc = desc.replace("</b>", "**")
            desc = desc.replace("<i>", "*")
            desc = desc.replace("</i>", "*")
            desc = desc.replace("<code>", "``")
            desc = desc.replace("</code>", "``")
            desc = desc.replace("<sup>", ":superscript:`")
            desc = desc.replace("</sup>", "`")
            desc = desc.replace("&amp;nbsp;", " ")
            desc = desc.replace("<ul>", "\n\n")
            desc = desc.replace("<li>", " * ")
            desc = desc.replace("</li>", "\n")
            desc = desc.replace("</ul>", "\n")
            desc = desc.replace("<br /><br />", " |br| ")
            desc = desc.replace("<br />", " |br| ")
            desc = desc.replace(" &amp;lt; ", " :math:`<` ")
            desc = desc.replace(" &amp;le; ", " :math:`\\leq` ")
            desc = desc.replace(" &amp;gt; ", " :math:`>` ")
            desc = desc.replace("&amp;lt;", "<")
            desc = desc.replace("&amp;gt;", ">")

            # Tidy up.
            desc = desc.strip()
            lines = desc.split("\n")
            return "\n".join("%s" % indent + x.strip() for x in lines)
    return ""


def get_default(node):
    type_node = get_first_child(node, ["type", "t"])
    default = get_attribute(type_node, ["default", "def", "d"], False)
    type_name = get_attribute(type_node, ["name", "t"])
    required = get_attribute(node, ["required"])
    if type_name is None:
        raise RuntimeError("Invalid setting")

    # If required make sure the default is blank!
    if required is not None and required == "TRUE":
        default = ""
    else:
        # If the default isn't set we have a problem!
        if default is None and "FILE" not in type_name:
            raise RuntimeError("Invalid setting, missing default!")

        # Format double type default values.
        if type_name in ("DOUBLE", "DOUBLERANGE"):
            default = format_double_string(default)

        elif type_name == "DOUBLERANGEEXT":
            # Check if the default string can be converted to a double
            try:
                value = float(default)
                default = format_double_string(default)
            except ValueError:
                options = get_type_params(type_node)
                if default not in options[2:]:
                    print(options[2:], default)
                    raise ValueError("Invalid default on DoubleRangeExt")

        elif type_name == "RANDOMSEED":
            try:
                value = int(default)
            except ValueError:
                if default.upper() != "TIME":
                    raise ValueError("Invalid default on RandomSeed")

        # Format option list defaults by expanding the first match
        elif type_name == "OPTIONLIST":
            options = get_type_params(type_node)
            for option in options:
                if option.startswith(default):
                    tmp = option.split()
                    default = tmp[0]
                    break

    return default


def get_allowed(node):
    type_node = get_first_child(node, ["type", "t"])
    type_name = get_attribute(type_node, ["name", "t"])
    if type_name is None:
        raise RuntimeError("Invalid setting")

    allowed = "FIXME"

    if type_name in "BOOL":
        return "Boolean ('true' or 'false')"
    elif type_name in ("DOUBLE", "INT", "STRING"):
        return type_name[0].upper() + type_name[1:].lower()
    elif type_name == "UINT":
        return "Unsigned integer"
    elif type_name == "INTPOSITIVE":
        return "Integer > 0"
    elif type_name == "UNSIGNEDDOUBLE":
        return "Unsigned double"
    elif type_name == "RANDOMSEED":
        return r"Integer :math:`\geq` 1, or 'time'"
    elif type_name == "INPUTFILELIST":
        return "CSV list of path names"
    elif type_name in ("INPUTFILE", "OUTPUTFILE", "INPUTDIRECTORY"):
        return "Path name"
    elif type_name == "INTLIST":
        return "CSV integer list"
    elif type_name == "DOUBLELIST":
        return "Double, or CSV list of doubles."
    elif type_name == "DATETIME":
        return "Double (if MJD), or formatted date-time string."
    elif type_name == "TIME":
        return "Double (if seconds), or formatted time string."
    elif type_name == "INTRANGE":
        p = get_type_params(type_node)
        if p[1] == "MAX":
            return r"Integer :math:`\geq` %s" % (p[0])
        else:
            return r"Integer in range %s :math:`\leq x \leq` %s" % (p[0], p[1])
    elif type_name == "INTLISTEXT":
        p = get_type_params(type_node)
        if p is None:
            raise ValueError("Invalid IntListExt specified")
        else:
            return "CSV integer list or '%s'" % p[0]
    elif type_name == "INTRANGEEXT":
        p = get_type_params(type_node)
        if p[1] == "MAX":
            allowed = r"Integer :math:`\geq` %s" % (p[0])
        else:
            allowed = r"Integer in range %s :math:`\leq x \leq` %s" % (p[0], p[1])
        if len(p) == 3:
            allowed += r" , or '%s'" % (p[2])
        if len(p) == 4:
            allowed += r" , '%s' or '%s'" % (p[2], p[3])
    elif type_name == "DOUBLERANGE":
        p = get_type_params(type_node)
        # TODO logic on not setting both values
        # for eg 2 to max display as >= 2
        # for eg min to 5 display as <= 5
        if p[1] == "MAX":
            allowed = r"Double :math:`\geq` %s" % (p[0])
        else:
            allowed = r"Double in range %s :math:`\leq x \leq` %s" % (p[0], p[1])
    elif type_name == "DOUBLERANGEEXT":
        p = get_type_params(type_node)
        if p[1] == "MAX":
            allowed = r"Double :math:`\geq` %s" % (p[0])
        else:
            allowed = r"Double in range %s :math:`\leq x \leq` %s" % (p[0], p[1])
        if len(p) == 3:
            allowed += r" , or '%s'" % (p[2])
        if len(p) == 4:
            allowed += r" , '%s' or '%s'" % (p[2], p[3])
    elif type_name == "OPTIONLIST":
        p = get_type_params(type_node)
        allowed = "One of the following:\n"
        for option in p:
            allowed += "\n  * " + option.strip()
        return allowed
    return allowed


def process_xml(root, node, lists, parent_key="", depth=0, count=0, file=None):
    # Loop over child nodes of the current node.
    for child in node:
        # Check type of node.
        if child.tag in ("s", "setting"):
            # Settings node.
            local_key = get_attribute(child, ["key", "k"], False)
            key = parent_key + "/" + local_key
            if not local_key:
                raise RuntimeError("Invalid settings node: has no key!")

            # Check for top-level group.
            type_node = child.find("type")
            if type_node is None and depth == 0:
                # Add to the settings toctree.
                key = local_key
                print("***** [GROUP %s] *****" % (key))
                section = "settings-%s" % key
                lists.append("   %s" % section)

                # Create a new reST file for the settings group.
                description = get_description(child, "")
                label_node = get_first_child(child, ["label"])
                label = get_node_text(label_node)
                file = open(
                    "@PROJECT_SOURCE_DIR@/docs/settings/%s.rst" % section, "w")
                file.write(".. Autogenerated file - do not edit!\n\n")
                file.write(".. |vspace| raw:: latex\n\n   \\vspace{5mm}\n\n")
                file.write(".. |br| raw:: html\n\n   <br /><br />\n\n\n")
                file.write(":tocdepth: 1\n\n")
                file.write(label + "\n")
                file.write("-"*len(label) + "\n\n")
                if description:
                    file.write("%s\n\n" % description)
                # Intermediate heading, to make the rest a bit smaller.
                file.write("List of settings keys\n")
                file.write("^^^^^^^^^^^^^^^^^^^^^\n\n")

            # Settings node.
            elif type_node is not None:
                count += 1
                print("[%03d]%s%s" % (count, " "*depth, key))
                description = get_description(child, "  ")
                default_value = get_default(child)
                heading = "``" + key + "``"
                if "required" in child.attrib:
                    heading += "  (Required)"
                file.write(heading + "\n")
                file.write("\"" * len(heading) + "\n")
                file.write("%s\n\n  " % description)
                if default_value:
                    file.write("**Default:** %s | " % default_value)
                file.write("**Allowed values:** %s" % get_allowed(child))
                file.write("\n\n")

            # Descend the tree.
            count = process_xml(
                root, child, lists, key, depth + 1, count, file)

            # Close the file on the last top-level setting.
            if depth == 0 and file:
                file.write("\n.. raw:: latex\n\n    \\clearpage\n\n")
                file.close()

        elif child.tag == "import":
            # Import group of tags.
            # Look for the "group" attribute (key to import).
            group = child.attrib.get("group")
            if group is None:
                continue

            import_group = get_node(root, group, "")
            if import_group is not None:
                count = process_xml(
                    root, import_group, lists, parent_key, depth, count, file)
            else:
                raise RuntimeError("Could not find group '%s'" % group)
    return count


def main():
    with open("@PROJECT_BINARY_DIR@/oskar/apps/xml/oskar_all.xml", "r") as f:
        xml_str = f.read()
    xml = ET.fromstring(xml_str)
    lists = []
    count = process_xml(xml, xml, lists)
    print("Processed %d settings" % count)

    # Create the parent settings file with the toctree filled in.
    with open("@PROJECT_SOURCE_DIR@/docs/settings/settings.rst.in", "r") as f:
        in_text = f.read()
    out_text = in_text.replace("SETTINGS_LISTS", "\n".join(lists))
    with open("@PROJECT_SOURCE_DIR@/docs/settings/settings.rst", "w") as f:
        f.write(out_text)


if __name__ == "__main__":
    main()
