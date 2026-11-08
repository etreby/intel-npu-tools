# RPM package.
#
# Built and installed on both Fedora and openSUSE Tumbleweed on every push, in
# containers, by the install-fedora and install-opensuse jobs. Each builds the
# spec on the distribution it targets — which is the only way the %if below is
# evaluated both ways — installs the result, checks that every dependency name
# it declares resolves there, and checks the installed tree.
#
# Enterprise Linux is still unverified: it needs EPEL for ffmpeg-free and
# tesseract, and nothing here has been run against it.
#
# Installs the same tree as the Debian and Arch packages by calling the shared
# staging script, so the three cannot drift apart.
#
# The models and the OpenVINO runtime are deliberately absent: they are large,
# they are redistributed under their own licences, and OpenVINO is not in the
# distribution archives. Each user runs intel-npu-tools-setup once, which is
# also why nothing here touches the network — a maintainer script must not.
#
# Build with:
#   rpmbuild -bb packaging/intel-npu-tools.spec \
#            --define "_sourcedir $PWD" --define "_projectdir $PWD"

%global project %{?_projectdir}%{!?_projectdir:%{_sourcedir}}
%global debug_package %{nil}

Name:           intel-npu-tools
Version:        0.3.0
Release:        1%{?dist}
Summary:        Local speech, OCR and semantic search on the Intel AI Boost NPU

License:        MIT
URL:            https://github.com/etreby/intel-npu-tools
BuildArch:      noarch

# The staging script rasterises the application icon at build time. Without
# this the build still succeeds and quietly ships the scalable icon alone,
# which is why render-icons.sh now treats a missing rasteriser as an error.
%if 0%{?suse_version}
BuildRequires:  rsvg-convert
%else
BuildRequires:  librsvg2-tools
%endif

# Dependency names differ between the RPM distributions, so they are selected
# rather than assumed. Fedora's names are not openSUSE's: ffmpeg-free,
# tesseract-langpack-eng and librsvg2-tools simply do not exist there, and a
# spec that names them fails to install on a distribution it claims to support.
# Both sets are now checked by name on the distribution they are for, which is
# what caught tesseract-ocr-traineddata-arabic: openSUSE names its language
# data by the three-letter code, and a Recommends that matches nothing is
# dropped in silence, so Arabic recognition would simply never have worked.
#
# ffmpeg and python3 are deliberately unversioned virtual names. openSUSE has
# no package called either, and requiring ffmpeg-7 or python311 instead would
# break at the next major version.
%if 0%{?rhel} == 9
# Enterprise Linux 9 ships 3.9 as python3 and will never ship anything newer
# under that name, so python3 >= 3.10 is not a dependency that can be
# satisfied there at all — it simply refuses to install. 3.12 is in AppStream
# beside it under its own name, and intel-npu-tools-setup picks the newest
# interpreter it can find rather than assuming python3 is new enough.
Requires:       python3.12
Requires:       python3.12-pip
%else
Requires:       python3 >= 3.10
%endif
Requires:       pciutils

%if 0%{?suse_version}
Requires:       ffmpeg
Requires:       tesseract-ocr
Requires:       tesseract-ocr-traineddata-english
Recommends:     tesseract-ocr-traineddata-ara
Recommends:     pipewire-tools
Recommends:     rsvg-convert
%else
Requires:       ffmpeg-free
Requires:       tesseract
Requires:       tesseract-langpack-eng
Recommends:     tesseract-langpack-ara
Recommends:     pipewire-utils
Recommends:     librsvg2-tools
%endif

Recommends:     wl-clipboard

Suggests:       gnome-screenshot
%if ! 0%{?rhel}
# Neither Enterprise Linux nor EPEL carries grim, and a suggestion naming a
# package that does not exist is dropped without a word. Guarded for every
# Enterprise Linux rather than for 9 alone, because a suggestion that is
# absent costs a wlroots user on a distribution that does not ship wlroots
# compositors nothing, and an untested release failing to build would cost
# rather more.
Suggests:       grim
%endif

%description
Makes the integrated Intel AI Boost NPU in Arrow Lake processors useful on
Linux. Provides private semantic search, local Whisper transcription,
screenshot text extraction, a control panel, and an MCP server that AI agents
can call to keep bulk text out of their context window.

The models and the OpenVINO runtime are not included, because they are large
and are redistributed under their own licences. Run intel-npu-tools-setup once
as your own user to create the environment and download them.

%prep
# Nothing to unpack: the package is built from the working tree.

%build
# Nothing to compile.

%install
%{project}/scripts/stage-package.sh %{buildroot} %{_prefix}

%files
# LICENSE is not marked as a license file here: it sits inside the lib
# directory this package already owns wholesale, and naming it again makes
# rpmbuild report the file twice. A copy is installed under the doc directory
# below. Note that rpm expands macros inside comments too, so this deliberately
# spells out paths in words rather than using them.
%doc %{_datadir}/doc/intel-npu-tools
%{_bindir}/intel-npu-info
%{_bindir}/intel-npu-mcp
%{_bindir}/intel-npu-ocr
%{_bindir}/intel-npu-panel
%{_bindir}/intel-npu-search
%{_bindir}/intel-npu-speech
%{_bindir}/intel-npu-transcribe
%{_bindir}/intel-npu-tools-setup
%{_prefix}/lib/intel-npu-tools
%{_datadir}/applications/intel-npu-speech.desktop
%{_datadir}/applications/intel-npu-ocr.desktop
%{_datadir}/applications/intel-npu-panel.desktop
%{_datadir}/icons/hicolor/*/apps/intel-npu-tools.*

%post
/bin/touch --no-create %{_datadir}/icons/hicolor &>/dev/null || :
cat <<'MESSAGE'

Intel NPU Tools is installed. Each user runs this once to create their
environment and download the models:

    intel-npu-tools-setup

MESSAGE

%postun
if [ $1 -eq 0 ] ; then
    /usr/bin/gtk-update-icon-cache %{_datadir}/icons/hicolor &>/dev/null || :
    /usr/bin/update-desktop-database &>/dev/null || :
fi
# A user's environment and models live in their home directory and are left
# alone: removing a package must not delete a user's data.

%posttrans
/usr/bin/gtk-update-icon-cache %{_datadir}/icons/hicolor &>/dev/null || :
/usr/bin/update-desktop-database &>/dev/null || :

%changelog
* Sun Aug 09 2026 Mohamed El-Etreby <40498+etreby@users.noreply.github.com> - 0.3.0-1
- Control panel, persistent settings, and desktop support beyond KDE
- context_filter and screen_to_text for reducing AI agent token use
- Optional cross-encoder reranking and a selectable speech model
