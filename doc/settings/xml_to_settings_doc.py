#!/usr/bin/python

# from xml.dom import minidom
#filename = 'oskar_simulator_2.xml'
# xmldoc = minidom.parse(filename)
#
# #print xmldoc.toxml()
#
# root = xmldoc.firstChild
# print len(root.childNodes)
# for i in range (0,len(root.childNodes)):
#     print '----------'
#     print root.childNodes[i].toxml()
#     print '----------'

import xml.etree.ElementTree as ET
filename = 'oskar_simulator_2.xml'
tree = ET.parse(filename)
root = tree.getroot()
print root.tag
print root.attrib

