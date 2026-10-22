-- Make a whole cross-reference, "Exhibit 3" or "Section 2.1", the PDF link.
--
-- In LaTeX output Quarto writes @tbl-x as the word, a no-break space and
-- \ref{tbl-x}, and only \ref is a link, so the colored link is the number
-- alone. On these pages "Exhibit 3" (this page's) has to stand apart from
-- "the paper's Table 3" (plain text), and a one-digit link is too easy to
-- miss. Runs after Quarto has resolved the references, and only for LaTeX.

local NBSP = '\u{a0}'

local function linked(word, label)
  return pandoc.RawInline('latex',
    '\\hyperref[' .. label .. ']{' .. word .. '~\\ref*{' .. label .. '}}')
end

-- '(Exhibit' gives '(' and 'Exhibit': punctuation before the word stays outside the link
local function split_word(text)
  return text:match('^(.-)(%a+)$')
end

function Inlines(inlines)
  if not FORMAT:match('latex') then return nil end
  local out = pandoc.Inlines({})
  for _, el in ipairs(inlines) do
    local label = el.t == 'RawInline' and el.text:match('^\\ref{([^}]+)}$')
    local n = #out
    local head, word, drop
    if label and n >= 2 and out[n].t == 'Str' and out[n].text == NBSP and out[n - 1].t == 'Str' then
      -- the word and the no-break space as separate strings
      head, word = split_word(out[n - 1].text)
      drop = 2
    elseif label and n >= 1 and out[n].t == 'Str' and out[n].text:sub(-#NBSP) == NBSP then
      -- the word and the no-break space merged into one string
      head, word = split_word(out[n].text:sub(1, -#NBSP - 1))
      drop = 1
    end
    if word then
      for _ = 1, drop do out:remove(#out) end
      if head ~= '' then out:insert(pandoc.Str(head)) end
      out:insert(linked(word, label))
    else
      out:insert(el)
    end
  end
  return out
end
