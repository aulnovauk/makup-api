{pkgs}: {
  deps = [
    pkgs.glib
    pkgs.libz
    pkgs.libGL
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
  ];
}
