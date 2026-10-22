--[[
  bold-table-headers.lua

  Pandoc sets LaTeX table headers in the body font. The booktabs rules separate
  the header from the body, but nothing distinguishes it typographically, and on
  a slide read from the back of a room that is not enough.

  Bolding by hand means writing | **Group** | **Limit** | ... in every table and
  remembering to do it in the next one. This does it for every table instead.
  Copied from aggregate\presentations into the house kit on 2026-10-18, where it
  runs for every house format; tables drawn by greater_tables are not pandoc
  tables and are untouched.

  LaTeX only. The HTML writer already emits <th>, which the theme bolds, so
  adding Strong there would be redundant markup for no visual change.
]]

local function bold_blocks(blocks)
  local out = {}
  for i, block in ipairs(blocks) do
    if block.t == 'Plain' then
      out[i] = pandoc.Plain { pandoc.Strong(block.content) }
    elseif block.t == 'Para' then
      out[i] = pandoc.Para { pandoc.Strong(block.content) }
    else
      -- A header cell holding anything else (a list, a nested block) is left
      -- alone rather than guessed at.
      out[i] = block
    end
  end
  return out
end

function Table(tbl)
  if not (FORMAT:match 'latex' or FORMAT:match 'beamer') then
    return nil
  end
  for _, row in ipairs(tbl.head.rows) do
    for _, cell in ipairs(row.cells) do
      cell.contents = bold_blocks(cell.contents)
    end
  end
  return tbl
end
