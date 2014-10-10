#!/usr/bin/python

import xml.etree.ElementTree as ET
import os
import re

# http://stackoverflow.com/questions/8845456/how-can-i-do-replace-a-child-elements-in-elementtree
def process_import_nodes(xmlFile):
    print ''
    print '** INFO: Reading xml file:', xmlFile,'**'
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

def get_attribute(node, allowed_keys, toupper=True):
    if node == None: return None
    for key in allowed_keys:
        if key in node.attrib:
            a = node.attrib[key]
            if toupper: return a.upper()
            else: return a
    return None



def get_first_child(node, possible_tags):
    if node == None: return None
    for child in node:
        for tag in possible_tags:
            if child.tag == tag:
                return child
    return None

def get_node_text(node):
    import re
    if node.text == None: return None
    pattern = re.compile(r'\s+')
    txt = node.text
    txt = re.sub(pattern, ' ', txt)
    return txt.strip()

def get_type_params(node):
    # TODO Ignore csv inside quotes
    # http://stackoverflow.com/questions/8208358/split-string-ignoring-delimiter-within-quotation-marks-python
    params_ = get_node_text(node)
    if params_ == None: return None
    return params_.split(',')

def format_double_string(string):
    if not 'e' in string:
        value = string.split('.')
        if len(value) == 1:
            return '%.1f' % float(string)
        else:
            return '%.*f' % (len(value[1]), float(string))

def set_default_value_latex(node, latex_file):
    type_      = get_first_child(node, ['type', 't'])
    default_   = get_attribute(type_, ['default', 'def', 'd'], False)
    type_name_ = get_attribute(type_, ['name', 't'])
    required_  = get_attribute(node, ['required'])
    if type_name_ == None: raise 'Invalid setting'

    # If required make sure the default is blank!
    if required_ != None and required_ == 'TRUE':
        default_ = ''
    else:
        # If the default isnt set we have a problem!
        if default_ == None and not 'FILE' in type_name_:
            raise RuntimeError('Invalid setting, missing default!')
    
        # Format double type default values.
        if type_name_ == 'DOUBLE' or type_name_ == 'DOUBLERANGE':
            default_ = format_double_string(default_)

        elif type_name_ == 'DOUBLERANGEEXT':
            # Check if the default string can be converted to a double
            try:
                value = float(default_)
                default_ = format_double_string(default_)
            except ValueError:
                options_ = get_type_params(type_)
                print options_[2:], default_
                if not default_ in options_[2:]:
                    raise RuntimeError('Invalid default on DoubleRangeExt')

        elif type_name_ == 'RANDOMSEED':
            try:
                value = int(default_)
            except ValueError:
                if default_.upper() != 'TIME':
                    raise RuntimeError('Invalid default on RandomSeed')

        # Format option list defaults by expanding the first match
        elif type_name_ == 'OPTIONLIST':
            options_ = get_type_params(type_)
            for option in options_:
                if option.startswith(default_):
                    tmp = option.split()
                    default_ = tmp[0]
                    break
            #print default_

    latex_file.write('%s\n' % default_)

def set_allowed_values_latex(node, latex_file):
    type_      = get_first_child(node, ['type', 't'])
    type_name_ = get_attribute(type_, ['name', 't'])
    if type_name_ == None: raise 'Invalid setting'

    allowed_values_ = 'Allowed values'

    if type_name_  == 'DOUBLE' or type_name_ == 'INT' or \
        type_name_ == 'BOOL' or type_name_ == 'STRING':
        allowed_values_ = type_name_[0].upper() + type_name_[1:].lower()

    elif type_name_ == 'UINT':
        allowed_values_ = 'Unsigned int'

    elif type_name_ == 'INTPOSITIVE':
        allowed_values_ = 'Integer > 0'

    elif type_name_  == 'UNSIGNEDDOUBLE':
        params_ = get_type_params(type_)
        allowed_values_ = 'Unsigned double'

    elif type_name_  == 'RANDOMSEED':
        allowed_values_ = "`time' or integer seed $\geq$ 1"

    elif type_name_  == 'DOUBLERANGE':
        params_ = get_type_params(type_)
        # TODO logic on not setting both values
        # for eg 2  to max display as  >= 2
        # for eg min to 5 display as  <= 5
        a = params_[0]
        b = params_[1]
        allowed_values_ = 'Double. %s $\leq$ value $\leq$ %s' % (a, b)

    elif type_name_  == 'DOUBLERANGEEXT':
        params_ = get_type_params(type_)
        min_ = params_[0]
        max_ = params_[1]
        special_value = params_[2]
        # TODO logic on range as for DoubleRange
        # eg  Double > 2 or max
        allowed_values_ = "Double in range %s $\leq$ value $\leq$ %s or `%s'" % (min_, max_, special_value)

    elif type_name_ == 'INPUTFILE' or type_name_ == 'INPUTFILELIST' or \
        type_name_ == 'OUTPUTFILE':
        allowed_values_ = 'Path name'
    
    latex_file.write('%s\n' % allowed_values_)
    latex_file.write('&\n')

def parse_setting_node(node, key, depth, count, latex_file=None):
    if not (node.tag == 's' or node.tag == 'setting'): return

    if 'key' in node.attrib:
        key_ = node.attrib['key']
    elif 'k' in node.attrib:
        key_ = node.attrib['k']
    else:
        raise 'Invalid settings node: has no key!'
    type_node_ = node.find('type')
    
    # =============== Top level group =========================================
    if type_node_ == None and depth == 1:
        #print '%s<G> key:%s' % ('  '*depth, key_)
        print ''
        print '***** [GROUP %s] *****' % (key_)

        html_file = '@PROJECT_SOURCE_DIR@/doc/settings/html_simulator_settings_text.html'
        filename_ = '@PROJECT_SOURCE_DIR@/doc/settings/latex_%s.tex' % (key_)
        dox_filename = '@PROJECT_SOURCE_DIR@/doc/settings/settings_tables.txt'

        label_ = ''
        for child_ in node:
            if child_.tag == 'label' or child_.tag == 'l':
                if child_.text != None:
                    label_ = child_.text
                    break
        gdesc_ = ''
        for child_ in node:
            if child_.tag == 'description' or child_.tag == 'desc':
                if child_.text != None:
                    gdesc_ = get_node_text(child_)
                    break

        # Create a section in the generated settings doxygen
        
        dox_file = open(dox_filename, 'a')
        dox_file.write('\n')
        dox_file.write(r'<!-- ************************************* -->' + '\n')
        dox_file.write(r'\subsection' + ' settings_' + key_ + ' ' + label_ +'\n')
        if len(gdesc_) > 0:
            dox_file.write(gdesc_ + '\n')
        dox_file.write('\n')
        dox_file.write(r'All settings keys in this group are prefixed with <tt>')
        dox_file.write(key_ + r'/</tt>.' + '\n')
        dox_file.write('\n')
        head, tail = os.path.split(filename_)
        dox_file.write(r'\latexinclude ' + tail + '\n')
        head, tail = os.path.split(html_file)
        dox_file.write(r'\htmlinclude  ' + tail + '\n')
        dox_file.close()

        # Create the latex table file for the settings group.
        # ----------------------------------------------------------------------
        
        print 'New table file: %s' % filename_
        latex_file_ = open(filename_,'w')

        latex_file_.write('%\n')
        latex_file_.write('% This is an autogenerated file - do not edit!\n')
        latex_file_.write('%\n')
        latex_file_.write('\n')
        #latex_file_.write(r'\renewcommand{\thefootnote}{\alph{footnote}}' + '\n')
        latex_file_.write('\n')
        latex_file_.write(r'\fontsize{8}{10}\selectfont' + '\n')
        #latex_file_.write(r'\fontsize{6}{10}\selectfont' + '\n')
        latex_file_.write(r'\begin{center}'+'\n')
        latex_file_.write(r'\begin{longtable}{|c|L{9cm}|L{8.5cm}|L{3.0cm}|L{1.7cm}|}' + '\n')
        #latex_file_.write(r'\begin{longtable}{|L{9cm}|L{8.5cm}|L{3.0cm}|L{1.7cm}|}' + '\n')
        latex_file_.write('\n')
        
        multi_header = True
        if multi_header == True:
            # Header for first page
            latex_file_.write(r'\hline'+'\n')
            latex_file_.write(r'  \rowcolor{lightgray}'+'\n')
            latex_file_.write(r'  {\textbf{ID}} &'+'\n')
            latex_file_.write(r'  {\textbf{Key}} &'+'\n')
            latex_file_.write(r'  {\textbf{Description}} &'+'\n')
            latex_file_.write(r'  {\textbf{Allowed values}} &'+'\n')
            latex_file_.write(r'  {\textbf{Default}} \\[0.5ex] \hline'+'\n')
            latex_file_.write(r'\endfirsthead'+'\n')
    
            # Header for remaining page(s)
            latex_file_.write(r'\hline'+'\n')
            latex_file_.write(r'  \rowcolor{lightgray}' + '\n')
            latex_file_.write(r'  {\textbf{ID}} &'+'\n')
            latex_file_.write(r'  {\textbf{Key}} &'+'\n')
            latex_file_.write(r'  {\textbf{Description}} &'+'\n')
            latex_file_.write(r'  {\textbf{Allowed values}} &'+'\n')
            latex_file_.write(r'  {\textbf{Default}} \\[0.5ex] \hline'+'\n')
            latex_file_.write(r'\endhead'+'\n')
    
            # Footer for all pages except the last page of the table
            latex_file_.write(r'  \multicolumn{5}{l}{{Continued on next page\ldots}} \\'+'\n')
            latex_file_.write(r'\endfoot'+'\n')
            # Footer for last page of the table
            latex_file_.write(r' '+'\n')
            latex_file_.write(r'\endlastfoot'+'\n')

        else:
            latex_file_.write('\hline\n')
            latex_file_.write(r'\rowcolor{lightgray}' + '\n')
            latex_file_.write(r'{\textbf{ID}}&' + '\n')
            latex_file_.write(r'{\textbf{Key}}&' + '\n')
            latex_file_.write(r'{\textbf{Description}}&' + '\n')
            latex_file_.write(r'{\textbf{Allowed values}}&' + '\n')
            latex_file_.write(r'{\textbf{Default}}' + '\n')
            latex_file_.write(r'\\' + '\n')
            latex_file_.write(r'\hline' + '\n')
            latex_file_.write('\n')

        # ----------------------------------------------------------------------
        return None, latex_file_
    
    # =============== Settings node ============================================
    else:
        if key: full_key = key + '/' + key_
        else: full_key = key_
        
        # >>>>> Key <<<<<<
        #print ''
        #print '%s<S> key:%s [%s]' % ('  '*depth, key_, full_key)
        if type_node_ == None:
            print '[---]%s%s' % (' '*depth, full_key)
        else:
            print '[%03i]%s%s' % (count, ' '*depth, full_key)
            
#             if len(full_key.split('/'))%2 == 1:
#                 latex_file.write(r'  \rowcolor{lightgray}'+'\n')
            
            latex_file.write('%03i\n' % count)
            latex_file.write('&\n')
            
            # escape '_' characters in the latex
            latex_key = full_key
#             if len(latex_key) > 50:
#                 current_length = 0
#                 print 'WRAPPING KEY', latex_key
#                 keys = latex_key.split('/')
#                 for i in range(0, len(keys)):
#                     current_length += len(keys[i])
#                     print key[i], current_length
#                     if i == len(keys)/2:
#                         latex_file.write(r'\newline' + '\n')
#                         latex_file.write(r'\indent ')
#                         current_length = 0
#                     latex_file.write(keys[i].replace('_', '\_'))
#                     if i < len(keys)-1:
#                         latex_file.write('/')
#                     else:
#                         latex_file.write('\n')
#             else:
            latex_key = latex_key.replace('_', '\_')
            #latex_file.write(latex_key + ' ' +str(len(latex_key)) +'\n')
            
            # TODO put footnote symbol for required keywords
            if 'required' in node.attrib:
                latex_file.write(latex_key + r'\textsuperscript{\textbf{\dag}}' + '\n')
                #latex_file.write(latex_key + r'\footnotemark[1]' + '\n')
                #latex_file.write('{{'+latex_key+r'}\tablenotemark{a}'+'}\n')
                #latex_file.write('{'+latex_key+'}\n')
            else:
                #latex_file.write('{'+latex_key+'}\n')
                latex_file.write(latex_key+'\n')
            latex_file.write('&\n')
    
            # >>>>> Description <<<<<
            for child_ in node:
                if child_.tag == 'desc' or child_.tag == 'd' or child_.tag == 'description':
                    # Extract the contents of the description tag as a string.
                    desc_ = ET.tostring(child_, method="xml")
                    desc_ = desc_.replace('<desc>', '')
                    desc_ = desc_.replace('</desc>', '')
                    desc_ = desc_.replace('<d>', '')
                    desc_ = desc_.replace('</d>', '')
                    desc_ = desc_.replace('<description>', '')
                    desc_ = desc_.replace('</description>', '')
#                     if child_.text == None:
#                         desc_ = ''
#                         continue
#                     else:
#                         desc_ = child_.text
#
#                     # If the description text contains xml entries,
#                     # these have to be appended correctly to the description.
#                     for format_tag in child_:
#                         desc_ += r'<' + format_tag.tag + r'>'
#                         desc_ += format_tag.text
#                         desc_ += r'</' + format_tag.tag + r'>'
#                         desc_ += format_tag.tail
# #                     num_format_tags = len(list(child_))
# #                     if num_format_tags >= 1:
# #                         print "text:[",child_.text,"]"
# #                         print "tail:[",child_.tail,"]"
# #                         print "desc:[",desc_,"]"

                    latex_desc_ = desc_
                    pattern = re.compile(r'\s+')
                    latex_desc_ = re.sub(pattern, ' ', latex_desc_)
                    latex_desc_ = latex_desc_.strip()
                    latex_desc_ = latex_desc_.replace(" '", " `")
                    latex_desc_ = latex_desc_.replace(r'&amp;lt;', r'$<$')
                    latex_desc_ = latex_desc_.replace(r'&amp;le;', r'$\leq$')
                    latex_desc_ = latex_desc_.replace(r'&amp;gt;', r'$>$')
                    latex_desc_ = latex_desc_.replace(r"<b>", r'\textbf{')
                    latex_desc_ = latex_desc_.replace(r"</b>", r'}')
#                     print latex_desc_
#                     print ''
                    latex_file.write('%s\n' % latex_desc_)
                    latex_file.write('&\n')
    
            # >>>>> Allowed values <<<<<<
            set_allowed_values_latex(node, latex_file)
    
            # >>>>> Default value <<<<<<
            set_default_value_latex(node, latex_file)
        
            # End the table row
            latex_file.write('\\\\\n')
            latex_file.write('\\hline\n')
            latex_file.write('\n')

        return full_key, latex_file

def recurse_tree(node, key=None, depth=0, count=0, latex_file=None):
    depth += 1

    # Loop over child nodes of the current node.
    for child_ in node:

        # Only handle settings nodes here.
        if child_.tag != 's': continue

        # Settings node
        key_, latex_file = parse_setting_node(child_, key, depth, count, latex_file)

        # Increment the setting count if a valid setting.
        if key_ != None:
            type_ = child_.find('type')
            if type_ != None:
                count+=1

        # Descend the tree.
        new_depth = depth+1
        count = recurse_tree(child_, key_, new_depth, count, latex_file)

        # For the last settings table have to close it here.
        #if depth == 1 and child_ == node[-1]:
        if depth == 1 and latex_file:
            print 'Closing latex table'
            #latex_file.write('\caption{Caption...}\n')
            #latex_file.write(r'\footnotetext[1]{Required setting.}'+'\n')
            latex_file.write(r'\end{longtable}'+'\n')
            latex_file.write(r'\end{center}'+'\n')
            latex_file.write(r'\normalsize'+'\n')
            #latex_file.write(r'\renewcommand{\thefootnote}{\arabic{footnote}}'+'\n')
            latex_file.write(r'\newpage'+'\n')
            latex_file.close()

    return count

filename = '@PROJECT_BINARY_DIR@/settings/xml/oskar.xml'
dox_filename = '@PROJECT_SOURCE_DIR@/doc/settings/settings_tables.txt'
if os.path.isfile(dox_filename):
    os.remove(dox_filename)


xml = process_import_nodes(filename)
#print ET.tostring(xml)
# print type(xml)
#print '---------------------------------'
#print '<%s>' % xml.tag
recurse_tree(xml)



# Create the settings.dox file.
file_replace = '@PROJECT_SOURCE_DIR@/doc/settings/settings_tables.txt'
file_in = '@PROJECT_SOURCE_DIR@/doc/settings/settings.dox.in'
file_out = '@PROJECT_SOURCE_DIR@/doc/settings/settings.dox'
# Get the text to insert
fh = open(file_replace, 'r')
replace_text = fh.read()
fh.close()
fh = open(file_in,'r')
in_text = fh.read()
fh.close()
out_text = in_text.replace('SETTINGS_TABLES', replace_text)
fh = open(file_out,'w')
fh.write(out_text)
fh.close()

