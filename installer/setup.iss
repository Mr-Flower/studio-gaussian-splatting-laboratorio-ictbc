; Programma di installazione di Studio Gaussian Splatting (Inno Setup 6.4 o successivo).
;
; Copia il programma nella cartella scelta, poi esegue installa.ps1, che scarica e configura i
; componenti (Python, PyTorch, nerfstudio, COLMAP, FFmpeg, MeshLab, codice Inria) mostrando
; l'avanzamento nella procedura guidata. Non richiede diritti di amministratore.
;
; Lo costruisce il workflow .github/workflows/release.yml. A mano, dalla radice del repository:
;   ISCC /DVersione=0.3.0 installer\setup.iss      ->  dist\GaussianSplatting-Setup-0.3.0.exe

#ifndef Versione
  #define Versione "0.0.0"
#endif
#define Nome "Studio Gaussian Splatting"
#define IdApp "LaboratorioICTBC.StudioGaussianSplatting"

[Setup]
AppId={{8E1B6C0E-5C0B-4C6F-9B6A-3A0F1D2C7E41}
AppName={#Nome}
AppVersion={#Versione}
AppVerName={#Nome} {#Versione}
AppPublisher=Laboratorio ICTBC
AppPublisherURL=https://github.com/Mr-Flower/studio-gaussian-splatting-laboratorio-ictbc
DefaultDirName={autopf}\Gaussian Splatting
DefaultGroupName=Gaussian Splatting
DisableProgramGroupPage=yes
DisableDirPage=no
; Installazione per il solo utente corrente: {autopf} e' %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
InfoBeforeFile=prima.txt
SetupIconFile=..\assets\icona.ico
UninstallDisplayIcon={app}\assets\icona.ico
UninstallDisplayName={#Nome}
OutputDir=..\dist
OutputBaseFilename=GaussianSplatting-Setup-{#Versione}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Spazio occupato dai componenti scaricati dopo la copia (ambiente Python e strumenti): circa 10 GB.
ExtraDiskSpaceRequired=10737418240
SetupLogging=yes
CloseApplications=no

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un'icona sul &desktop"; GroupDescription: "Icone:"

[Files]
Source: "..\app\*.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "..\assets\icona.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "..\installa.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
; Moduli CUDA gia' compilati: se sono accanto al sorgente entrano nel Setup, altrimenti installa.ps1 li scarica.
Source: "..\tools\wheels\*.whl"; DestDir: "{app}\tools\wheels"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\tools\wheels\LICENZE-moduli-compilati.txt"; DestDir: "{app}\tools\wheels"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#Nome}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app"; WorkingDir: "{app}"; IconFilename: "{app}\assets\icona.ico"; AppUserModelID: "{#IdApp}"
Name: "{group}\Cartella di lavoro"; Filename: "{code:CartellaLavoro}"
Name: "{group}\Completa o ripara l'installazione"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -NoExit -File ""{app}\installa.ps1"" -SenzaCollegamento"; WorkingDir: "{app}"
Name: "{group}\Disinstalla {#Nome}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#Nome}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app"; WorkingDir: "{app}"; IconFilename: "{app}\assets\icona.ico"; AppUserModelID: "{#IdApp}"; Tasks: desktopicon

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m app"; WorkingDir: "{app}"; Description: "Avvia {#Nome}"; Flags: nowait postinstall skipifsilent; Check: ComponentiInstallati

[UninstallDelete]
; Scaricati o creati dopo la copia dei file. La cartella di lavoro (progetti e risultati) non viene toccata.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: files; Name: "{app}\cartella_lavoro.txt"

[Code]
var
  PaginaLavoro: TInputDirWizardPage;
  Riuscita: Boolean;
  Errore: String;

procedure InitializeWizard;
begin
  PaginaLavoro := CreateInputDirPage(wpSelectDir,
    'Cartella di lavoro', 'Dove salvare progetti e risultati?',
    'Progetti, modelli creati e report vengono salvati in questa cartella. Un progetto con centinaia di ' +
    'foto occupa decine di gigabyte: conviene un disco capiente e una cartella non sincronizzata in rete ' +
    '(OneDrive, Dropbox).', False, '');
  PaginaLavoro.Add('');
  PaginaLavoro.Values[0] := GetPreviousData('CartellaLavoro', ExpandConstant('{%USERPROFILE}\Gaussian Splatting'));
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'CartellaLavoro', PaginaLavoro.Values[0]);
end;

function CartellaLavoro(Param: String): String;
begin
  Result := RemoveBackslash(PaginaLavoro.Values[0]);
end;

function ComponentiInstallati: Boolean;
begin
  Result := Riuscita;
end;

function ComandoInstalla(Script, Argomenti: String): String;
begin
  Result := '-NoProfile -ExecutionPolicy Bypass -File "' + Script + '" ' + Argomenti;
end;

{ installa.ps1 scrive "== [3/6] Titolo del passo" a ogni passo e "ERRORE: ..." se si ferma. }
procedure RigaDiInstalla(const S: String; const Error, FirstLine: Boolean);
var
  Testo: String;
begin
  Testo := Trim(S);
  if Testo = '' then
    Exit;
  if Pos('ERRORE:', Testo) = 1 then
    Errore := Trim(Copy(Testo, 8, Length(Testo)))
  else if Pos('== [', Testo) = 1 then
  begin
    WizardForm.StatusLabel.Caption := 'Passo ' + Copy(Testo, 5, Pos('/', Testo) - 5) + ' di ' +
      Copy(Testo, Pos('/', Testo) + 1, Pos(']', Testo) - Pos('/', Testo) - 1) + ': ' +
      Copy(Testo, Pos(']', Testo) + 2, Length(Testo));
    WizardForm.FilenameLabel.Caption := '';
  end
  else
    WizardForm.FilenameLabel.Caption := Testo;
end;

{ Verifica computer e cartella con gli stessi controlli di installa.ps1, prima di copiare qualcosa. }
function ControlliSuperati(Cartella: String): Boolean;
var
  Codice: Integer;
begin
  ExtractTemporaryFile('installa.ps1');
  Errore := '';
  Result := ExecAndLogOutput('powershell.exe',
    ComandoInstalla(ExpandConstant('{tmp}\installa.ps1'), '-SoloControlli -Cartella "' + Cartella + '"'),
    '', SW_HIDE, ewWaitUntilTerminated, Codice, @RigaDiInstalla) and (Codice = 0);
  if (not Result) and (Errore = '') then
    Errore := 'Non è stato possibile verificare i requisiti del computer (PowerShell non disponibile).';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    if not ControlliSuperati(RemoveBackslash(WizardDirValue)) then
    begin
      SuppressibleMsgBox(Errore, mbError, MB_OK, IDOK);
      Result := False;
    end;
  end
  else if CurPageID = PaginaLavoro.ID then
  begin
    if Trim(PaginaLavoro.Values[0]) = '' then
    begin
      SuppressibleMsgBox('Indicare la cartella di lavoro.', mbError, MB_OK, IDOK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Codice: Integer;
  Righe: TArrayOfString;
begin
  if CurStep <> ssPostInstall then
    Exit;
  ForceDirectories(CartellaLavoro(''));
  SetArrayLength(Righe, 1);
  Righe[0] := CartellaLavoro('');
  SaveStringsToUTF8File(ExpandConstant('{app}\cartella_lavoro.txt'), Righe, False);

  WizardForm.StatusLabel.Caption := 'Scarico e configuro i componenti: può richiedere da 10 a 40 minuti.';
  WizardForm.FilenameLabel.Caption := '';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Errore := '';
  Riuscita := ExecAndLogOutput('powershell.exe',
    ComandoInstalla(ExpandConstant('{app}\installa.ps1'), '-SenzaCollegamento'),
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Codice, @RigaDiInstalla) and (Codice = 0);
  WizardForm.ProgressGauge.Style := npbstNormal;
  WizardForm.FilenameLabel.Caption := '';
  if not Riuscita then
  begin
    if Errore = '' then
      Errore := 'lo script di installazione si è interrotto (codice ' + IntToStr(Codice) + ').';
    SuppressibleMsgBox('Il programma è stato copiato, ma non è stato possibile scaricare e configurare tutti i ' +
      'componenti:' + #13#10#13#10 + Errore + #13#10#13#10 +
      'Controllare la connessione a Internet e scegliere «Completa o ripara l''installazione» dal menu Start: ' +
      'riprende da dove si è fermata.', mbError, MB_OK, IDOK);
  end;
end;
