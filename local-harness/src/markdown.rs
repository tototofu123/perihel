//! Minimal Markdown renderer for the harness body.
//!
//! It turns model output into ANSI-styled, already-wrapped lines. The layout
//! tests in `main.rs` require every returned line to fit the given width, so
//! wrapping happens here with display-column widths (CJK aware) and all
//! escape sequences are emitted after wrapping.

use crate::char_width;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Style {
    Plain,
    Bold,
    Italic,
    Code,
    Dim,
    Heading,
    Quote,
}

impl Style {
    fn ansi(self) -> &'static str {
        match self {
            Style::Plain => "",
            Style::Bold => "\x1b[1m",
            Style::Italic => "\x1b[3m",
            Style::Code => "\x1b[38;5;222m",
            Style::Dim => "\x1b[2m",
            Style::Heading => "\x1b[1;38;5;39m",
            Style::Quote => "\x1b[38;5;244m",
        }
    }
}

struct Span {
    style: Style,
    text: String,
}

impl Span {
    fn new(style: Style, text: impl Into<String>) -> Self {
        Span { style, text: text.into() }
    }
}

fn styled(style: Style, spans: Vec<Span>) -> Vec<Span> {
    spans.into_iter().map(|s| Span { style, text: s.text }).collect()
}

/// Render markdown into styled lines no wider than `width` display columns.
pub fn render(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out: Vec<String> = Vec::new();
    let mut code: Option<(String, Vec<String>)> = None;

    for raw in text.lines() {
        let line = raw.trim_end();
        let trimmed = line.trim_start();

        if trimmed.starts_with("```") {
            match code.take() {
                Some((lang, lines)) => flush_code(&lang, &lines, width, &mut out),
                None => {
                    let lang = trimmed.trim_start_matches('`').trim().to_lowercase();
                    code = Some((lang, Vec::new()));
                }
            }
            continue;
        }

        if let Some((_, lines)) = code.as_mut() {
            lines.push(line.to_string());
            continue;
        }

        if trimmed.is_empty() {
            out.push(String::new());
            continue;
        }

        if is_rule(trimmed) {
            out.push(format!("\x1b[2m{}\x1b[0m", "-".repeat(width.min(40))));
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('#') {
            let title = rest.trim_start_matches('#').trim();
            let spans = styled(Style::Heading, parse_inline(title));
            push_wrapped(&spans, width, &mut out);
            continue;
        }

        if trimmed.starts_with('>') {
            let body = trimmed.trim_start_matches('>').trim();
            let mut spans = vec![Span::new(Style::Quote, "| ")];
            spans.extend(parse_inline(body));
            push_wrapped(&spans, width, &mut out);
            continue;
        }

        if let Some((indent, body)) = list_item(trimmed) {
            let mut spans = vec![Span::new(Style::Dim, format!("{indent}- "))];
            spans.extend(parse_inline(body));
            push_wrapped(&spans, width, &mut out);
            continue;
        }

        let spans = parse_inline(trimmed);
        push_wrapped(&spans, width, &mut out);
    }

    if let Some((lang, lines)) = code {
        flush_code(&lang, &lines, width, &mut out);
    }

    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn is_rule(s: &str) -> bool {
    s.len() >= 3 && s.chars().all(|c| c == '-' || c == '*' || c == '_')
}

/// `- item`, `* item`, `1. item` -> ("", body) or ("  ", body) for nested.
fn list_item(s: &str) -> Option<(String, &str)> {
    let indent_len = s.len() - s.trim_start().len();
    let body = s.trim_start();
    let rest = body
        .strip_prefix("- ")
        .or_else(|| body.strip_prefix("* "))
        .or_else(|| body.strip_prefix("+ "));
    if let Some(rest) = rest {
        return Some((" ".repeat(indent_len.min(4)), rest));
    }
    // ordered list: "12. text"
    let digits: usize = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 {
        let tail = &body[digits..];
        if let Some(rest) = tail.strip_prefix(". ") {
            return Some((" ".repeat(indent_len.min(4)), rest));
        }
    }
    None
}

fn flush_code(lang: &str, lines: &[String], width: usize, out: &mut Vec<String>) {
    let pad = 2usize;
    let body_width = width.saturating_sub(pad).max(4);
    if lines.is_empty() {
        out.push("\x1b[48;5;236m\x1b[0m".to_string());
        return;
    }
    for line in lines {
        let clipped = clip_display(line, body_width);
        let used = display_width(&clipped) + pad;
        let fill = width.saturating_sub(used);
        let mut s = String::from("\x1b[48;5;236m");
        s.push_str(&" ".repeat(pad));
        s.push_str(&highlight(lang, &clipped));
        s.push_str(&" ".repeat(fill));
        s.push_str("\x1b[0m");
        out.push(s);
    }
}

fn push_wrapped(spans: &[Span], width: usize, out: &mut Vec<String>) {
    out.extend(wrap_spans(spans, width));
}

fn wrap_spans(spans: &[Span], width: usize) -> Vec<String> {
    let mut words: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    for span in spans {
        for ch in span.text.chars() {
            if ch.is_whitespace() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else {
                cur.push((ch, span.style));
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }

    let mut lines = Vec::new();
    let mut line: Vec<(char, Style)> = Vec::new();
    let mut line_w = 0usize;

    for word in words {
        let word_w: usize = word.iter().map(|(c, _)| char_width(*c)).sum();
        if word_w > width {
            for (c, st) in word {
                let cw = char_width(c);
                if line_w + cw > width && !line.is_empty() {
                    lines.push(emit(&line));
                    line.clear();
                    line_w = 0;
                }
                line.push((c, st));
                line_w += cw;
            }
            continue;
        }
        let extra = if line.is_empty() { 0 } else { 1 };
        if !line.is_empty() && line_w + extra + word_w > width {
            lines.push(emit(&line));
            line.clear();
            line_w = 0;
        }
        if !line.is_empty() {
            line.push((' ', Style::Plain));
            line_w += 1;
        }
        line_w += word_w;
        line.extend(word);
    }

    if !line.is_empty() {
        lines.push(emit(&line));
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn emit(chars: &[(char, Style)]) -> String {
    let mut out = String::new();
    let mut current = Style::Plain;
    for (c, style) in chars {
        if *style != current {
            out.push_str("\x1b[0m");
            out.push_str(style.ansi());
            current = *style;
        }
        out.push(*c);
    }
    out.push_str("\x1b[0m");
    out
}

fn parse_inline(s: &str) -> Vec<Span> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut plain = String::new();
    let mut i = 0usize;

    let flush = |plain: &mut String, out: &mut Vec<Span>| {
        if !plain.is_empty() {
            out.push(Span::new(Style::Plain, std::mem::take(plain)));
        }
    };

    while i < chars.len() {
        let c = chars[i];

        if c == '`' {
            if let Some(end) = find_char(&chars, i + 1, '`') {
                flush(&mut plain, &mut out);
                out.push(Span::new(Style::Code, chars[i + 1..end].iter().collect::<String>()));
                i = end + 1;
                continue;
            }
        }

        if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if let Some(end) = find_seq(&chars, i + 2, "**") {
                flush(&mut plain, &mut out);
                out.push(Span::new(Style::Bold, chars[i + 2..end].iter().collect::<String>()));
                i = end + 2;
                continue;
            }
        }

        if (c == '*' || c == '_') && !chars.get(i + 1).map_or(true, |n| n.is_whitespace()) {
            if let Some(end) = find_char(&chars, i + 1, c) {
                flush(&mut plain, &mut out);
                out.push(Span::new(Style::Italic, chars[i + 1..end].iter().collect::<String>()));
                i = end + 1;
                continue;
            }
        }

        if c == '[' {
            if let Some(close) = find_char(&chars, i + 1, ']') {
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(paren) = find_char(&chars, close + 2, ')') {
                        flush(&mut plain, &mut out);
                        let label: String = chars[i + 1..close].iter().collect();
                        let url: String = chars[close + 2..paren].iter().collect();
                        out.push(Span::new(Style::Plain, label));
                        out.push(Span::new(Style::Dim, format!(" ({url})")));
                        i = paren + 1;
                        continue;
                    }
                }
            }
        }

        plain.push(c);
        i += 1;
    }

    flush(&mut plain, &mut out);
    if out.is_empty() {
        out.push(Span::new(Style::Plain, String::new()));
    }
    out
}

fn find_char(chars: &[char], from: usize, needle: char) -> Option<usize> {
    (from..chars.len()).find(|&i| chars[i] == needle && (i == 0 || chars[i - 1] != '\\'))
}

fn find_seq(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let target: Vec<char> = needle.chars().collect();
    if from >= chars.len() {
        return None;
    }
    (from..chars.len()).find(|&i| {
        i + target.len() <= chars.len() && chars[i..i + target.len()] == target[..]
    })
}

fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn clip_display(s: &str, cols: usize) -> String {
    let mut out = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = char_width(c);
        if w + cw > cols {
            break;
        }
        out.push(c);
        w += cw;
    }
    out
}

const KEYWORDS: &[&str] = &[
    // rust / typescript-ish
    "fn", "let", "mut", "pub", "use", "struct", "enum", "impl", "match", "if", "else", "for",
    "while", "loop", "return", "async", "await", "move", "trait", "where", "const", "static",
    "mod", "crate", "self", "Self", "super", "type", "unsafe", "extern", "dyn", "ref", "in",
    "as", "break", "continue", "true", "false", "None", "Some", "Ok", "Err", "function", "var",
    "new", "this", "typeof", "instanceof", "export", "default", "extends", "null", "undefined",
    "switch", "case", "throw", "catch", "do", "of",
    // python
    "def", "class", "import", "from", "pass", "raise", "try", "except", "finally", "with",
    "lambda", "yield", "global", "nonlocal", "assert", "del", "elif", "and", "or", "not", "is",
    "True", "False",
    // shell
    "echo", "then", "fi", "esac", "local", "readonly", "exit", "set", "unset", "cd", "export",
];

fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word)
}

/// Very small highlighter: keywords, strings, comments and numbers.
fn highlight(lang: &str, line: &str) -> String {
    let comment = match lang {
        "py" | "python" | "sh" | "bash" | "shell" | "zsh" | "yaml" | "yml" | "toml" | "rb"
        | "ruby" | "ps1" | "powershell" | "make" | "dockerfile" => Some("#"),
        "sql" => Some("--"),
        "" => None,
        _ => Some("//"),
    };

    // diff blocks get line-level colouring instead of token scanning.
    if matches!(lang, "diff" | "patch") {
        let colour = if line.starts_with('+') {
            "\x1b[38;5;114m"
        } else if line.starts_with('-') {
            "\x1b[38;5;203m"
        } else if line.starts_with("@@") {
            "\x1b[38;5;81m"
        } else {
            "\x1b[38;5;252m"
        };
        return format!("{colour}{line}\x1b[0m");
    }

    let chars: Vec<char> = line.chars().collect();
    let mut out = String::new();
    let mut i = 0usize;

    while i < chars.len() {
        // comment to end of line
        if let Some(tok) = comment {
            let t: Vec<char> = tok.chars().collect();
            if chars.len() - i >= t.len() && chars[i..i + t.len()] == t[..] {
                out.push_str("\x1b[38;5;244m");
                out.extend(chars[i..].iter());
                out.push_str("\x1b[0m");
                return out;
            }
        }

        let c = chars[i];

        if c == '"' || c == '\'' || c == '`' {
            let quote = c;
            let start = i;
            i += 1;
            while i < chars.len() {
                if chars[i] == '\\' {
                    i += 2;
                    continue;
                }
                if chars[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            let end = i.min(chars.len());
            out.push_str("\x1b[38;5;114m");
            out.extend(chars[start..end].iter());
            out.push_str("\x1b[0m");
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '.')
            {
                i += 1;
            }
            out.push_str("\x1b[38;5;215m");
            out.extend(chars[start..i].iter());
            out.push_str("\x1b[0m");
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if is_keyword(&word) {
                out.push_str("\x1b[38;5;81m");
                out.push_str(&word);
                out.push_str("\x1b[0m");
            } else {
                out.push_str(&word);
            }
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn width_of(line: &str) -> usize {
        // count display width ignoring ANSI
        let mut w = 0;
        let mut it = line.chars().peekable();
        while let Some(c) = it.next() {
            if c == '\x1b' {
                if it.peek() == Some(&'[') {
                    it.next();
                    for d in it.by_ref() {
                        if ('\x40'..='\x7e').contains(&d) {
                            break;
                        }
                    }
                }
                continue;
            }
            w += char_width(c);
        }
        w
    }

    #[test]
    fn every_line_fits_the_width() {
        let text = "# Title\n\nSome **bold** and *italic* with `inline code` and a [link](https://example.com/a/very/long/url).\n\n- first item that is fairly long and will need wrapping at narrow widths\n- second item\n\n```rust\nfn main() { println!(\"hello world this is a long line\"); } // comment\n```\n\n> a quote block\n\n---\n\n日本語のテキストも混ざる場合があります。";
        for w in 12..=100 {
            for line in render(text, w) {
                assert!(width_of(&line) <= w, "width {w}: {line:?} is {} wide", width_of(&line));
            }
        }
    }

    #[test]
    fn inline_styles_are_applied_and_markers_removed() {
        let lines = render("a **b** `c` *d*", 60);
        let joined = lines.join("\n");
        assert!(joined.contains("\x1b[1m"), "bold missing");
        assert!(joined.contains("\x1b[38;5;222m"), "inline code missing");
        assert!(joined.contains("\x1b[3m"), "italic missing");
        assert!(!joined.contains('*'), "bold markers left: {joined:?}");
        assert!(!joined.contains('`'), "code markers left: {joined:?}");
    }

    #[test]
    fn fenced_block_gets_a_background_and_highlighting() {
        let lines = render("```rust\nfn main() {}\n```", 40);
        assert!(lines.iter().all(|l| l.contains("\x1b[48;5;236m")), "no block background: {lines:?}");
        assert!(lines.iter().any(|l| l.contains("\x1b[38;5;81m")), "no keyword colour: {lines:?}");
        let plain: String = lines.join("");
        assert!(!plain.contains("```"), "fence leaked into output");
    }

    #[test]
    fn unterminated_fence_still_renders() {
        let lines = render("```python\ndef f():\n    return 1", 40);
        assert!(lines.iter().all(|l| l.contains("\x1b[48;5;236m")));
        assert!(lines.iter().any(|l| l.contains("38;5;81")));
    }
}
