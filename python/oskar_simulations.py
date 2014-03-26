

class Settings:
    def __ini__(self, filename)
        self.filename = filename
        self.group = []

    def beginGroup(self, name):
        self.group.append(name)

    def endGroup(self):


    def set(self, key, value):
        set_(key, value)

    def set_(self, key, value):
        import os
        os.system('oskar_settings_set -q '+filename+' '+key+' "'+str(value)+'"')
