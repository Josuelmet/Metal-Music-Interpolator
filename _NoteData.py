class NoteData:
    
    def __init__(self, value=None, duration=None, isDotted=None, palmMute=False):
        self.value = value
        self.duration = duration
        self.isDotted = isDotted
        self.palmMute = palmMute
        
    def as_tuple(self):
        return (self.value, self.duration, self.isDotted, self.palmMute)
        
    def __str__(self):
        return f"{self.value},\t{'dotted ' if self.isDotted else ''} 1/{int(self.duration)}, {'muted' if self.palmMute else ''}"
        