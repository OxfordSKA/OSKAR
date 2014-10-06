#!/usr/bin/python

import xml.etree.ElementTree as ET
import os

# http://stackoverflow.com/questions/8845456/how-can-i-do-replace-a-child-elements-in-elementtree
def process_import_nodes(xmlFile):
    xmlFH = open(xmlFile, 'r')
    xmlStr = xmlFH.read()
    et = ET.fromstring(xmlStr)
    parent_map = dict((c, p) for p in et.getiterator() for c in p)
    # ref: http://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node/2170994
    importList = et.findall('.//import[@filename]')
    for importPlaceholder in importList:
        old_dir = os.getcwd()
        new_dir = os.path.dirname(importPlaceholder.attrib['filename'])
        shallPushd = os.path.exists(new_dir)
        if shallPushd:
            print "  pushd: %s" %(new_dir)
            os.chdir(new_dir) # pushd (for relative linking)

        # Recursing to import element from file reference
        importedElement = process_import_nodes(os.path.basename(importPlaceholder.attrib['filename']))

        # element replacement
        parent = parent_map[importPlaceholder]
        index = parent._children.index(importPlaceholder)
        parent._children[index] = importedElement

        if shallPushd:
            print "  popd: %s" %(old_dir)
            os.chdir(old_dir) # popd

    return et

def parse_setting_node(node, key, depth):
    if not (node.tag == 's' or node.tag == 'setting'): return

    if 'key' in node.attrib:
        key_ = node.attrib['key']
    else:
        raise 'Invalid settings node: has no key!'
    type_node_ = node.find('type')
    if type_node_ == None and depth == 1:
        print '%s<G> key:%s' % ('  '*depth, key_)
        return None
    else:
        if key: full_key = key + '/' + key_
        else: full_key = key_
        print ''
        print '%s<S> key:%s [%s]' % ('  '*depth, key_, full_key)
        return full_key

def parse_type_node(node, depth):
    if not (node.tag == 'type' or node.tag == 't'): return
    attributes = node.attrib
    if 'name' in attributes:
        name_ = attributes['name']
    else:
        raise 'Invalid type tag!'
    if 'default' in attributes:
        default_ = attributes['default']
    else:
        default_ = ''
    print '%s<type> name:%s default:%s' % ('  '*(depth), name_, default_)

def parse_description_node(node, depth):
    if not (node.tag == 'description' or node.tag == 'desc'): return
    print '%s<Desc> %s' % ('  '*(depth), node.text)

def recurse_tree(root, key = None, depth=0):
    depth += 1
    for node_ in root:

        # Settings node
        new_key = parse_setting_node(node_, key, depth)

        # type node
        parse_type_node(node_, depth)

        # Description node
        parse_description_node(node_, depth)

        # Descend the tree.
        new_depth = depth+1
        recurse_tree(node_, new_key, new_depth)


filename = '@PROJECT_BINARY_DIR@/settings/xml/oskar.xml'


xml = process_import_nodes(filename)
# print ET.tostring(xml)
# print type(xml)
print '---------------------------------'
print '<%s>' % xml.tag
recurse_tree(xml)

