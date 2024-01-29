def i09_export_XPS(pathname, filenames):
    if length(varargin)==0:
        [filenames,pathname] = uigetfile({'*.nxs','nexus'},'MultiSelect','on');
    elif length(varargin)==1:
        pathname = varargin{1};
        cd(pathname)
        [filenames,pathname] = uigetfile({'*.nxs','nexus'},'MultiSelect','on');
    elif length(varargin)==2:
        filenames = varargin{2};
        pathname = varargin{1};
    else:
        display('i09_export_XPS requires a maximum of two inputs')
        return